# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    # =============================================================================
    # Dice Loss - 分割重叠度损失函数
    # 目标: 直接衡量预测mask与真实mask的重叠程度，关注分割的准确性
    # 优势: 1) 对目标尺寸不敏感(小物体和大物体权重相等)
    #      2) 直接优化分割评估指标 
    #      3) 提供区域级别的监督信号
    # 公式: Dice Loss = 1 - (2×交集 + 1) / (预测面积 + 真实面积 + 1)
    # =============================================================================
    
    # 将网络输出的logits转换为(0,1)区间的概率值
    # sigmoid函数: σ(x) = 1/(1+e^(-x))，将任意实数映射到概率空间
    inputs = inputs.sigmoid()
    
    if loss_on_multimask:
        # 多mask预测模式: inputs和targets形状为[N, M, H, W]，M对应多个预测mask
        assert inputs.dim() == 4 and targets.dim() == 4
        # 保持多mask维度，只展平空间维度(H,W) -> (H*W)
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        # 计算分子: 2 × 交集面积(预测概率×真实标签的像素级乘积)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        # 单mask模式: 直接展平所有空间维度
        inputs = inputs.flatten(1)
        # 计算分子: 2 × 交集面积
        numerator = 2 * (inputs * targets).sum(1)
    
    # 计算分母: 预测区域面积 + 真实区域面积
    denominator = inputs.sum(-1) + targets.sum(-1)
    
    # Dice Loss = 1 - Dice系数
    # 添加平滑项+1防止除零错误，当交集和并集都为0时loss=0
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    # =============================================================================
    # Sigmoid Focal Loss - 困难样本聚焦损失函数  
    # 目标: 解决前景/背景像素数量极度不平衡的问题，让模型重点关注困难像素
    # 
    # 核心思想: 
    # 1) α平衡因子: 解决正负样本数量不平衡(背景像素>>前景像素)
    #    α=0.25表示正样本权重25%，负样本权重75%
    # 2) γ聚焦因子: 降低简单样本权重，突出困难样本
    #    γ=2时，简单样本(预测置信度高)权重接近0，困难样本权重保持
    #
    # 实际效果:
    # - 背景像素(简单样本): 模型很容易预测正确 → focal loss大幅降低权重 → 梯度小
    # - 边界像素(困难样本): 模型难以预测 → focal loss保持较高权重 → 重点学习
    # =============================================================================
    
    # 将logits转换为预测概率
    prob = inputs.sigmoid()
    
    # 计算标准二元交叉熵损失，不进行reduction以便后续加权
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # 计算模型预测正确的概率p_t:
    # 当target=1时，p_t = prob (预测为正类的概率)
    # 当target=0时，p_t = 1-prob (预测为负类的概率)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    
    # Focal Loss核心: 用(1-p_t)^γ项动态调整样本权重
    # 当p_t接近1(预测很自信且正确)时，(1-p_t)^γ接近0，权重很小
    # 当p_t接近0(预测不自信或错误)时，(1-p_t)^γ接近1，权重较大  
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # α平衡因子: 为正负样本分配不同权重
        # 正样本(target=1): alpha_t = α
        # 负样本(target=0): alpha_t = 1-α
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        # 多mask模式: loss形状为[N, M, H, W]，M对应多个预测mask
        assert loss.dim() == 4
        # 在空间维度上取平均，保持batch和mask维度
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    # =============================================================================
    # IoU Loss - IoU质量预测损失函数
    # 目标: 训练网络预测自己输出mask的质量分数(IoU)
    # 
    # 作用:
    # 1) 让网络学会自我评估输出质量，知道哪个预测更准确
    # 2) 用于后处理时的置信度排序和最优mask选择  
    # 3) 支持多mask输出时的质量感知选择
    #
    # 实现机制:
    # 1) 计算预测mask与真实mask的实际IoU(使用硬分类)
    # 2) 与网络预测的IoU分数比较，使用L1或MSE损失优化
    # =============================================================================
    
    assert inputs.dim() == 4 and targets.dim() == 4
    
    # 将预测概率转换为硬分类mask(>0即为前景)
    # 注意: 这里使用硬阈值而非soft概率，因为IoU需要明确的区域边界
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    
    # 计算实际IoU = 交集面积 / 并集面积
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()  # 交集: 预测正确的前景像素
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()  # 并集: 所有前景像素
    # 使用clamp防除零，当并集为0时IoU定义为0
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    # 计算预测IoU与实际IoU之间的损失
    if use_l1_loss:
        # L1损失: |pred_iou - actual_iou|，对异常值更鲁棒
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        # MSE损失: (pred_iou - actual_iou)²，默认选择，梯度更平滑
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
        
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
    ):
        """
        This class computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """
        # =============================================================================
        # MultiStepMultiMasksAndIous - 综合损失管理器: 协调 focal、dice、iou、class四种损失
        # 目标: 管理和组合多种损失函数，支持多步预测和多mask输出
        #
        # 核心功能:
        # 1) 损失权重管理: 协调focal、dice、iou、class四种损失的权重
        # 2) 多步骤累积: 支持迭代式预测，累计每步的损失  
        # 3) 最优mask选择: 多mask预测时选择最佳mask进行反向传播
        # 4) 有效对象过滤: 只在有目标对象存在时才计算损失
        #
        # 损失权重配置示例:
        # weight_dict = {
        #     'loss_mask': 20,  # focal loss权重(主要)
        #     'loss_dice': 1,   # dice loss权重(辅助) 
        #     'loss_iou': 1,    # iou loss权重
        #     'loss_class': 1   # 对象存在性分类权重
        # }
        # =============================================================================

        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )  # Number of objects is fixed within a batch
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        # =============================================================================
        # 多步骤多mask损失计算的核心逻辑
        # 1) 从outputs中提取多步预测的mask、IoU分数、对象存在性分数
        # 2) 遍历每个预测步骤，累计所有损失
        # 3) 使用reduce_loss按权重合并最终损失
        # =============================================================================

        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4  # [N, 1, H, W]
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        # 初始化损失累加器 - 累计每个预测步骤的损失
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        # 计算加权总损失作为最终的core_loss用于反向传播
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        # =============================================================================
        # 单步损失更新函数 - 计算当前预测步骤的所有损失并累加
        # 核心机制:
        # 1) 计算所有类型的损失(focal、dice、iou、class)
        # 2) 多mask预测时选择focal+dice loss最小的mask
        # 3) 只在有目标对象存在时才计算损失(避免空点击的负样本)
        # =============================================================================
        
        # 将目标mask扩展为与预测mask相同的形状[N, M, H, W]
        target_masks = target_masks.expand_as(src_masks)
        
        # ========== 计算各种损失 ==========
        # 1) Focal Loss: 解决类别不平衡，关注困难像素
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        
        # 2) Dice Loss: 优化分割重叠度，保证分割质量  
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        
        # 3) Object Classification Loss: 预测点击位置是否存在对象
        if not self.pred_obj_scores:
            # 不预测对象存在性时，class loss为0，target_obj全为1
            loss_class = torch.tensor(
                0.0, dtype=loss_multimask.dtype, device=loss_multimask.device
            )
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            # 检查target_masks中是否真的有前景像素(>0)
            # target_obj=1表示存在对象，target_obj=0表示纯背景
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            # 使用focal loss训练对象存在性分类器
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        # 4) IoU Loss: 训练网络预测输出质量
        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        
        # ========== 多mask选择策略 ==========
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        
        if loss_multimask.size(1) > 1:
            # 多mask预测时，选择focal+dice loss最小的mask进行反向传播
            # 这种策略确保只对最有希望的预测进行优化，避免混淆
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            
            # IoU损失策略: 可选择监督所有mask或只监督最佳mask
            if self.supervise_all_iou:
                # 所有mask的IoU损失取平均(鼓励所有输出都有好的质量预测)
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                # 只监督最佳mask的IoU损失(与SAM保持一致)
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            # 单mask预测时直接使用
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # ========== 有效对象过滤 ==========
        # 关键机制: 只在有目标对象存在时才进行反向传播
        # 这避免了在纯背景区域点击时的误导性监督信号
        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        # ========== 损失累加 ==========
        # 累加到总损失中(注意损失已经按num_objects归一化)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        # =============================================================================
        # 损失加权合并函数
        # 根据weight_dict中的权重配置，将各种损失按比例合并为最终损失
        # 
        # 典型权重配置:
        # Total Loss = 20×Focal + 1×Dice + 1×IoU + 1×Class
        # 
        # 设计理念:
        # - Focal Loss权重最高(20): 主要负责解决类别不平衡和困难样本学习
        # - Dice Loss权重中等(1): 辅助优化分割质量和区域完整性  
        # - IoU/Class Loss权重较低(1): 提供质量预测和对象检测的补充监督
        # =============================================================================
        
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss
