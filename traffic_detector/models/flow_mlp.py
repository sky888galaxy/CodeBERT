"""流量行为检测的 MLP 模型实现。

模型基于通用流量特征空间进行恶意/正常判别，不依赖具体数据集，只依赖输入维度。
可选多分类头用于扩展攻击类型识别。
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """线性残差块，适配相同维度的 skip connection。"""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.seq(x)


class FlowMLPDetector(nn.Module):
    """流量行为检测的主力模型。

    结构围绕通用流量特征向量构建：多层全连接 + BN + GELU/Dropout，
    插入 1-2 个残差块增强表达力。模型只依赖输入维度，不绑定具体数据集。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        num_attack_types: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128, 64]

        modules: List[nn.Module] = []
        in_dim = input_dim
        for idx, h in enumerate(hidden_dims):
            modules.append(nn.Linear(in_dim, h))
            modules.append(nn.BatchNorm1d(h))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
            # 在偶数索引层加入一个同维度残差块，强调“加法”增益而非改特征空间。
            if h == in_dim and idx % 2 == 1:
                modules.append(ResidualBlock(h, dropout))
            in_dim = h

        self.backbone = nn.Sequential(*modules)

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.binary_head = nn.Linear(last_dim, 1)
        self.attack_type_head = (
            nn.Linear(last_dim, num_attack_types) if num_attack_types else None
        )

    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
        """前向计算。

        Args:
            x: [batch, input_dim] 的特征张量。
            return_logits: 是否直接返回 logits（便于配合 BCEWithLogitsLoss）。
        Returns:
            logits 或 概率字典。binary_head 始终存在，attack_type_head 可选。
        """

        out = x
        for layer in self.backbone:
            out = layer(out)

        binary_logits = self.binary_head(out)
        attack_logits = self.attack_type_head(out) if self.attack_type_head else None

        if return_logits:
            return binary_logits, attack_logits

        binary_prob = torch.sigmoid(binary_logits)
        attack_prob = torch.softmax(attack_logits, dim=-1) if attack_logits is not None else None
        return {"binary_prob": binary_prob, "attack_type_prob": attack_prob}


__all__ = ["FlowMLPDetector"]
