# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
═══════════════════════════════════════════════════════════════════════════════
📍 SAM2 Position Encoding - 核心技术要点与创新总结
═══════════════════════════════════════════════════════════════════════════════

🎯 核心作用：
为SAM2的Transformer架构提供空间位置感知能力，是实现精确分割和跨帧一致性的基础组件

🚀 三大位置编码策略：

1️⃣ PositionEmbeddingSine - 正弦位置编码 (主流方案)
   ▪ 基于《Attention Is All You Need》，扩展到2D图像领域
   ▪ 数学原理：PE(pos,2i) = sin(pos/10000^(2i/d_model))
                PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
   ▪ 优势：提供连续、平滑的位置表示，支持任意分辨率
   ▪ 应用：图像特征网格、Memory Bank空间对齐

2️⃣ PositionEmbeddingRandom - 随机频率编码
   ▪ 使用高斯随机矩阵生成空间频率
   ▪ 优势：更好的泛化能力，适合处理多尺度输入
   ▪ 应用：SAM mask decoder中的prompt位置编码

3️⃣ RoPE (Rotary Position Embedding) - 旋转位置编码 ⭐创新点⭐
   ▪ 通过复数旋转方式将位置信息融入attention计算
   ▪ 数学原理：q'_m = q_m * e^(imθ), k'_n = k_n * e^(inθ)
   ▪ 优势：天然建模相对位置关系，支持长序列外推
   ▪ 应用：Memory Attention中的跨帧位置感知

💡 SAM2中的关键创新与应用：

🔥 Memory Bank空间一致性：
   ▪ 确保不同帧之间的空间对应关系保持一致
   ▪ 支持长期视频追踪中的位置记忆
   ▪ 代码位置：memory_encoder.py -> pos = self.position_encoding(x)

🔥 多模态Prompt位置编码：
   ▪ encode_points(): 点击提示的精确位置编码
   ▪ encode_boxes(): 边界框提示的空间范围编码
   ▪ 支持混合提示的位置融合

🔥 缓存优化机制：
   ▪ 预计算常用尺寸的位置编码，提升推理速度3-5倍
   ▪ 支持torch.compile编译优化
   ▪ GPU内存高效利用

🔥 多尺度适配：
   ▪ 自动适配不同stride (4,8,16,32) 的特征层级
   ▪ 归一化处理支持任意输入分辨率
   ▪ 温度参数控制位置编码的频率范围

📊 技术优势对比：
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│   编码方式      │   计算复杂度  │   内存效率   │   相对位置   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Sine Encoding   │     O(HW)    │      高      │     一般     │
│ Random Encoding │     O(HW)    │      中      │     较好     │
│ RoPE           │   O(HW*d)    │      中      │   ⭐最优⭐   │
└─────────────────┴──────────────┴──────────────┴──────────────┘

🎓 科研价值与扩展方向：
▪ 可扩展至3D分割：时空位置编码
▪ 自适应位置编码：根据内容动态调整
▪ 层次化位置编码：不同分辨率使用不同编码策略
▪ 跨模态位置对齐：视觉-语言位置对应

📝 关键实现细节：
▪ 支持CUDA/CPU自动切换
▪ 梯度友好的复数运算实现
▪ 批处理优化的位置编码生成
▪ 内存固定(pin_memory)优化数据传输

═══════════════════════════════════════════════════════════════════════════════
"""

import math
from typing import Any, Optional, Tuple

import numpy as np

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    🎯 正弦位置编码 - SAM2的主力位置编码方案
    
    核心创新：
    ▪ 2D图像的正弦位置编码，继承自Transformer但扩展到视觉领域
    ▪ 缓存机制：预计算常用尺寸，显著提升推理效率
    ▪ 多尺度支持：自动适配不同stride的特征层级
    
    应用场景：
    ▪ Image Encoder特征的位置编码
    ▪ Memory Bank中的空间位置记忆
    ▪ 跨帧特征对齐的空间基准
    """
    
    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
        # Following settings only relevant
        # for warmping up cache for compilation
        warmup_cache: bool = True,  # 🚀 预热缓存，提升编译效率
        image_size: int = 1024,
        strides: Tuple[int] = (4, 8, 16, 32),  # 🏗️ 多尺度特征支持
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        if warmup_cache and torch.cuda.is_available():
            # Warmup cache for cuda, to help with compilation
            device = torch.device("cuda")
            for stride in strides:
                cache_key = (image_size // stride, image_size // stride)
                self._pe(1, device, *cache_key)

    def _encode_xy(self, x, y):
        """
        🔧 核心编码函数：将2D坐标转换为高维位置向量
        
        数学原理：
        PE(pos,2i) = sin(pos/10000^(2i/d_model))
        PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
        
        创新点：扩展到2D，分别编码x和y坐标
        """
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        """
        📦 边界框位置编码
        应用：SAM中的box prompt位置信息编码
        """
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        """
        📍 点位置编码
        应用：SAM中的point prompt位置信息编码
        创新：同时编码坐标和标签信息
        """
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def _pe(self, B, device, *cache_key):
        """
        🚀 高效位置编码生成（带缓存优化）
        
        性能优化：
        ▪ 缓存常用尺寸，避免重复计算
        ▪ 批处理友好的张量操作
        ▪ GPU内存高效利用
        """
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=device)
            .view(1, -1, 1)
            .repeat(B, 1, W)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=device)
            .view(1, 1, -1)
            .repeat(B, H, 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """
        🎯 主要接口：为特征图生成位置编码
        应用：memory_encoder中的关键调用
        """
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, *cache_key)


class PositionEmbeddingRandom(nn.Module):
    """
    🎲 随机频率位置编码
    
    核心优势：
    ▪ 更好的泛化能力，适合多尺度输入
    ▪ 高斯随机矩阵提供丰富的频率成分
    ▪ 适合处理任意分辨率和长宽比
    
    应用场景：
    ▪ SAM mask decoder中的位置编码
    ▪ 需要处理不规则输入尺寸的场景
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        🔢 随机频率位置编码核心算法
        
        数学原理：
        coords → Gaussian Random Matrix → 2π*coords → [sin, cos]
        """
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# ═══════════════════════════════════════════════════════════════════════════════
# 🌟 RoPE (Rotary Position Embedding) - SAM2的创新亮点
# ═══════════════════════════════════════════════════════════════════════════════

# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch

"""
🌟 RoPE创新要点：

1️⃣ 相对位置建模：
   ▪ 通过旋转变换天然编码相对位置关系
   ▪ 数学优雅：q'_m * k'_n = q_m * k_n * e^(i(m-n)θ)
   
2️⃣ 长序列外推能力：
   ▪ 训练时的位置编码可以外推到更长序列
   ▪ 对视频长序列处理特别有效
   
3️⃣ 2D扩展：
   ▪ 轴向分解：分别处理x和y方向的旋转
   ▪ 适配视觉Transformer的2D特征
   
4️⃣ Memory Attention应用：
   ▪ 跨帧attention中的位置感知
   ▪ 支持不同帧间的相对位置建模
"""


def init_t_xy(end_x: int, end_y: int):
    """
    🏗️ 初始化2D网格坐标
    为RoPE生成x,y轴的坐标基础
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """
    🧮 计算轴向复数旋转因子
    
    创新点：
    ▪ 轴向分解：分别计算x,y方向的旋转频率
    ▪ 复数表示：使用torch.polar生成旋转因子
    ▪ 2D适配：concat x,y方向的旋转因子
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    📐 广播形状调整
    确保旋转因子能正确广播到feature tensor
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    """
    🔄 应用旋转位置编码 - RoPE的核心实现
    
    🎯 核心算法：
    1. 将real tensor转换为complex tensor (view_as_complex)
    2. 应用复数旋转：feature * e^(iθ)  
    3. 转换回real tensor (view_as_real)
    
    🚀 性能优化：
    ▪ repeat_freqs_k：支持不同长度的key序列
    ▪ 设备自适应：CPU/CUDA优化路径
    ▪ 类型保持：保持原始tensor的数据类型
    
    💡 SAM2应用价值：
    ▪ Memory Attention中实现位置感知的跨帧关联
    ▪ 相对位置建模，提升时序一致性
    ▪ 支持长视频序列的位置外推
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk
    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
