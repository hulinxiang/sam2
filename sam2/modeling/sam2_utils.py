# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
═══════════════════════════════════════════════════════════════════════════════
🛠️ SAM2 核心工具函数库 - 架构支持与交互式分割工具集
═══════════════════════════════════════════════════════════════════════════════

🎯 文件定位：
SAM2系统的基础设施工具库，提供Memory Bank管理、交互式分割、网络组件等核心功能

🚀 主要功能模块：

1️⃣ Memory Bank 时序管理 ⭐核心创新⭐
   ▪ select_closest_cond_frames(): 智能历史帧选择策略
   ▪ 实现时序局部性原理，平衡性能与内存使用
   ▪ 支持长视频处理，避免GPU OOM

2️⃣ 交互式分割智能算法 ⭐重要创新⭐
   ▪ sample_random_points_from_errors(): 错误驱动的随机采样
   ▪ sample_one_point_from_error_center(): 基于距离变换的中心采样
   ▪ sample_box_points(): 噪声边界框点生成
   ▪ 自动识别预测错误，生成智能改进建议

3️⃣ 网络架构基础组件
   ▪ DropPath: 随机深度训练，提升泛化能力
   ▪ MLP: 标准多层感知机实现
   ▪ LayerNorm2d: 2D特征图的层归一化
   ▪ get_clones(): Transformer多层结构构建

4️⃣ 位置编码与激活函数工具
   ▪ get_1d_sine_pe(): 1D正弦位置编码生成
   ▪ get_activation_fn(): 统一激活函数接口

💡 技术创新要点：
🔥 时序局部性：优先选择时间接近的条件帧，体现视频连续性
🔥 错误驱动采样：基于FP/FN区域自动生成改进策略
🔥 距离变换优化：选择最稳定的采样点，提升交互质量
🔥 模块化设计：统一接口，支持灵活的策略切换

📊 应用场景：
▪ Video Object Segmentation中的帧间关联
▪ 交互式分割的智能点击建议
▪ Transformer架构的模块化构建
▪ 训练过程的正则化优化

═══════════════════════════════════════════════════════════════════════════════
"""

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    🎯 Memory Bank智能帧选择策略 - SAM2的核心创新算法
    
    核心作用：
    ▪ 从历史条件帧中选择最相关的帧用于当前预测
    ▪ 实现时序局部性原理：时间上接近的帧更重要
    ▪ 内存优化：限制参与attention的帧数，避免GPU OOM
    
    选择策略（智能优先级）：
    1️⃣ 最高优先级：frame_idx之前最近的一帧（历史连续性）
    2️⃣ 次高优先级：frame_idx之后最近的一帧（未来信息）
    3️⃣ 补充策略：按时间距离排序，选择其他最近帧
    
    技术优势：
    ▪ 时序感知：保持视频的时间连续性
    ▪ 双向考虑：同时利用过去和未来信息
    ▪ 渐进扩展：优雅地处理帧数限制
    
    应用场景：
    ▪ 长视频序列的Memory Bank管理
    ▪ Real-time推理的性能优化
    ▪ 跨帧attention的计算负载均衡
    
    参数说明：
    - frame_idx: 当前帧索引
    - cond_frame_outputs: 所有条件帧的输出dict
    - max_cond_frame_num: 最大条件帧数量（-1表示无限制）
    
    返回值：
    - selected_outputs: 选中的条件帧
    - unselected_outputs: 未选中的条件帧
    """
    
    # Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    # that are temporally closest to the current frame at `frame_idx`. Here, we take
    # - a) the closest conditioning frame before `frame_idx` (if any);
    # - b) the closest conditioning frame after `frame_idx` (if any);
    # c) any other temporally closest conditioning frames until reaching a total
    #      of `max_cond_frame_num` conditioning frames.

    # Outputs:
    # - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    # - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.

    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


def get_1d_sine_pe(pos_inds, dim, temperature=10000):
    """
    📏 1D正弦位置编码生成器
    
    核心功能：
    ▪ 生成经典的Transformer位置编码
    ▪ 基于《Attention Is All You Need》的数学公式
    ▪ 适用于序列位置信息编码
    
    应用场景：
    ▪ Object Pointer的时间位置编码
    ▪ 序列数据的位置感知
    ▪ 时序建模的基础组件
    
    数学原理：
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    # Get 1D sine positional embedding as in the original Transformer paper.
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed


def get_activation_fn(activation):
    """
    🔧 激活函数统一接口
    
    功能：为不同模块提供标准化的激活函数
    支持：relu, gelu, glu等主流激活函数
    """
    # Return an activation function given a string
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module, N):
    """
    🏗️ 模块复制器 - Transformer架构构建工具
    
    核心作用：
    ▪ 创建N个相同的模块副本
    ▪ 用于构建Transformer的多层结构
    ▪ 确保每层参数独立，避免共享权重
    
    应用场景：
    ▪ MemoryAttention的多层堆叠
    ▪ Encoder/Decoder层的复制
    ▪ 网络深度扩展的标准方法
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """
    🎲 随机深度训练 (Stochastic Depth) - 正则化创新技术
    
    核心原理：
    ▪ 训练时随机跳过某些层的前向传播
    ▪ 等价于动态调整网络深度
    ▪ 提升模型泛化能力，防止过拟合
    
    技术优势：
    ▪ 正则化效果：减少过拟合风险
    ▪ 训练加速：部分路径被跳过
    ▪ 模型鲁棒性：适应不同深度的网络
    
    应用场景：
    ▪ 深层Transformer的训练优化
    ▪ Vision Transformer的正则化
    ▪ 大模型训练的稳定性提升
    
    实现来源：huggingface/pytorch-image-models (timm)
    """
    
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    """
    🧠 多层感知机 (Multi-Layer Perceptron) - 标准前馈网络
    
    核心特性：
    ▪ 可配置的层数和隐藏维度
    ▪ 灵活的激活函数选择
    ▪ 可选的sigmoid输出层
    
    设计模式：
    input → Linear → Activation → ... → Linear → (Sigmoid)
    
    应用场景：
    ▪ Object Score Prediction：预测目标存在概率
    ▪ Feature Projection：特征维度变换
    ▪ Classification Head：分类任务的输出层
    
    技术来源：MaskFormer transformer predictor的轻量改版
    """
    
    # Lightly adapted from
    # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    """
    🔄 2D层归一化 - 视觉Transformer的关键组件
    
    核心创新：
    ▪ 针对2D特征图的层归一化实现
    ▪ 在channel维度进行归一化操作
    ▪ 适配ConvNet和ViT的混合架构
    
    技术优势：
    ▪ 训练稳定性：规范化特征分布
    ▪ 收敛加速：减少内部协变量偏移
    ▪ 性能提升：提高模型表达能力
    
    应用场景：
    ▪ Memory Encoder中的特征规范化
    ▪ ConvNeXt架构的核心组件
    ▪ 2D特征图的标准化处理
    
    实现来源：detectron2 → ConvNeXt的层归一化实现
    """
    
    # From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
    # Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    📦 边界框点采样器 - 交互式分割的输入生成
    
    核心功能：
    ▪ 从mask生成边界框的角点坐标
    ▪ 添加合理噪声模拟真实用户点击
    ▪ 提供标准化的box prompt格式
    
    算法流程：
    1️⃣ mask → bounding box坐标
    2️⃣ 提取左上角和右下角点
    3️⃣ 添加噪声模拟点击误差
    4️⃣ 边界限制确保坐标有效
    
    噪声策略：
    ▪ 噪声幅度 = min(bbox_size * noise, noise_bound)
    ▪ 平衡真实性和有效性
    ▪ 避免过大噪声导致无效点击
    
    应用场景：
    ▪ 训练数据增强：生成多样化的box prompts
    ▪ 交互模拟：测试系统对不精确输入的鲁棒性
    ▪ 自动标注：批量生成边界框标注
    
    标签约定：
    - 2: 左上角点 (top_left)
    - 3: 右下角点 (bottom_right)
    """
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )  # uncentered pixel coords
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

    box_coords = box_coords.reshape(-1, 2, 2)  # always 2 points
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """
    🎯 错误驱动的随机点采样 - 交互式分割的智能改进策略
    
    💡 核心创新：基于预测错误自动生成改进建议
    
    算法逻辑：
    1️⃣ False Positive (FP) 分析：~gt_masks & pred_masks
       ↳ 模型错误预测为前景的区域 → 采样负点纠正
    
    2️⃣ False Negative (FN) 分析：gt_masks & ~pred_masks  
       ↳ 模型遗漏的前景区域 → 采样正点补充
    
    3️⃣ 完全正确情况：gt_masks == pred_masks
       ↳ 从背景区域采样负点进行验证
    
    技术优势：
    ▪ 错误导向：专注于模型薄弱环节
    ▪ 自动化：无需人工分析错误类型
    ▪ 高效性：直接针对问题区域改进
    ▪ 泛化性：适用于任意分割任务
    
    采样策略：
    ▪ 使用随机噪声在FP/FN区域中采样
    ▪ 通过argmax选择最优采样位置
    ▪ 自动确定点击标签（0=负点，1=正点）
    
    应用场景：
    ▪ 交互式分割的迭代改进
    ▪ Active Learning的样本选择
    ▪ 模型错误分析和可视化
    ▪ 自动化的质量控制流程
    """
    if pred_masks is None:  # if pred_masks is not provided, treat it as empty
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks
    # whether the prediction completely match the ground-truth on each mask
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # channel 0 is FP map, while channel 1 is FN map
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    # sample a negative new click from FP region or a positive new click
    # from FN region, depend on where the maximum falls,
    # and in case the predictions are all correct (no FP or FN), we just
    # sample a negative click from the background region
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """
    🎖️ 错误中心点采样 - 基于距离变换的精确采样策略
    
    💡 核心创新：选择距离错误边界最远的点，确保采样稳定性
    
    算法核心：距离变换 (Distance Transform)
    1️⃣ 计算每个错误区域内各点到边界的距离
    2️⃣ 选择距离最大的点作为采样位置
    3️⃣ 比较FP和FN区域的最大距离值
    4️⃣ 在影响最大的区域进行采样
    
    技术优势：
    ▪ 稳定性：中心点不易受边界噪声影响
    ▪ 有效性：距离边界远的点修正效果更好
    ▪ 精确性：基于几何中心的数学严格性
    ▪ 鲁棒性：对不规则区域形状适应良好
    
    实现细节：
    ▪ 使用OpenCV的cv2.distanceTransform
    ▪ DIST_L2: 欧氏距离计算
    ▪ padding: 边界填充处理避免边缘效应
    ▪ 设备适配：CPU/GPU灵活切换
    
    理论基础：
    基于RITM (Reviving Iterative Training with Mask Guidance)的采样方法
    论文链接：https://github.com/saic-vul/ritm_interactive_segmentation
    
    应用场景：
    ▪ 高质量交互式分割
    ▪ 精确的错误修正
    ▪ 专业标注工具
    ▪ 医学影像分割等高精度场景
    """
    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    # false positive region, a new point sampled in this region should have
    # negative label to correct the FP error
    fp_masks = ~gt_masks & pred_masks
    # false negative region, a new point sampled in this region should have
    # positive label to correct the FN error
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        # compute the distance of each point in FN/FP region to its boundary
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        # take the point in FN/FP region with the largest distance to its boundary
        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im  # x
        points[b, 0, 1] = pt_idx // W_im  # y
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(gt_masks, pred_masks, method):
    """
    🎮 统一点采样接口 - 策略模式的优雅实现
    
    核心作用：
    ▪ 提供统一的点采样策略选择接口
    ▪ 支持不同采样方法的无缝切换
    ▪ 便于实验对比和算法评估
    
    支持策略：
    ▪ "uniform": 随机采样策略 (sample_random_points_from_errors)
    ▪ "center": 中心采样策略 (sample_one_point_from_error_center)
    
    设计优势：
    ▪ 策略模式：易于扩展新的采样方法
    ▪ 统一接口：简化上层调用逻辑
    ▪ 配置驱动：通过参数控制采样行为
    
    应用场景：
    ▪ 交互式分割系统的核心调度
    ▪ 不同采样策略的性能对比
    ▪ 算法研究的实验平台
    """
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")
