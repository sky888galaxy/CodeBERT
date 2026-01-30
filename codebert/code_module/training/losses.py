"""
多任务损失函数模块

包含：
1. FocalLoss - 处理类不平衡
2. WeightedCrossEntropyLoss - 加权交叉熵
3. MultiTaskLoss - 多任务损失合并
4. 自适应类权重计算
5. 样本权重机制（强/弱标注区分）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss - 重新加权简单样本和困难样本
    
    公式: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    适用于：类别严重不平衡的二分类或多分类任务
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        eps: float = 1e-7
    ):
        """
        Args:
            alpha: 类别平衡权重，形状 [num_classes]
                   若为 None，则不使用 alpha
                   示例：二分类 alpha=[0.25, 0.75] 表示负类权重 0.25，正类权重 0.75
            gamma: 难度系数，通常为 2.0 或 2.5
                   gamma 越大，对困难样本的关注越多
            reduction: 'mean' / 'sum' / 'none'
            eps: 防止 log(0) 的小值
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, num_classes]
        targets: torch.Tensor,  # [batch_size]
        sample_weights: Optional[torch.Tensor] = None  # [batch_size]
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 logits，形状 [batch_size, num_classes]
            targets: 目标标签，形状 [batch_size]
            sample_weights: 每个样本的权重（用于强/弱标注区分），形状 [batch_size]
        
        Returns:
            loss: 标量损失值
        """
        # 获取概率
        p = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        
        # 获取 target 对应的概率
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [batch_size]
        p_t = torch.gather(p, dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # 计算 focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss  # [batch_size]
        
        # 应用 alpha 平衡权重
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)  # [batch_size]
            focal_loss = alpha_t * focal_loss
        
        # 应用样本权重（强/弱标注）
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
        
        # 聚合
        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵损失 - 处理类别不平衡
    
    支持：
    1. 类别权重（自动或手动设置）
    2. 样本权重（强/弱标注）
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """
        Args:
            class_weights: 类别权重，形状 [num_classes]
                           示例：[0.3, 0.7] 表示类 0 权重 0.3，类 1 权重 0.7
            reduction: 'mean' / 'sum' / 'none'
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,  # [batch_size, num_classes]
        targets: torch.Tensor,  # [batch_size]
        sample_weights: Optional[torch.Tensor] = None  # [batch_size]
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出 logits
            targets: 目标标签
            sample_weights: 每个样本的权重（强/弱标注）
        
        Returns:
            loss: 标量损失值
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(
            logits, targets, 
            weight=self.class_weights,
            reduction='none'
        )  # [batch_size]
        
        # 应用样本权重
        if sample_weights is not None:
            ce_loss = ce_loss * sample_weights
        
        # 聚合
        if self.reduction == 'mean':
            loss = ce_loss.mean()
        elif self.reduction == 'sum':
            loss = ce_loss.sum()
        else:
            loss = ce_loss
        
        return loss


class MultiTaskLoss(nn.Module):
    """
    多任务损失 - 将二分类和类型分类损失合并
    
    公式: L = L_binary * w1 + L_type * w2
    
    支持：
    1. 对 type_label 无效的样本自动 mask
    2. 强/弱标注权重
    3. FocalLoss 或 CrossEntropy 切换
    """
    
    def __init__(
        self,
        binary_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.5,
        binary_class_weights: Optional[torch.Tensor] = None,
        type_class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            binary_loss_weight: 二分类损失权重
            type_loss_weight: 类型分类损失权重
            use_focal_loss: 是否使用 Focal Loss（否则用 CrossEntropy）
            focal_gamma: Focal Loss 的 gamma 参数
            binary_class_weights: 二分类的类权重
            type_class_weights: 类型分类的类权重
        """
        super(MultiTaskLoss, self).__init__()
        self.binary_loss_weight = binary_loss_weight
        self.type_loss_weight = type_loss_weight
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.binary_loss_fn = FocalLoss(
                alpha=binary_class_weights,
                gamma=focal_gamma,
                reduction='none'
            )
            self.type_loss_fn = FocalLoss(
                alpha=type_class_weights,
                gamma=focal_gamma,
                reduction='none'
            )
        else:
            self.binary_loss_fn = WeightedCrossEntropyLoss(
                class_weights=binary_class_weights,
                reduction='none'
            )
            self.type_loss_fn = WeightedCrossEntropyLoss(
                class_weights=type_class_weights,
                reduction='none'
            )
    
    def forward(
        self,
        binary_logits: torch.Tensor,        # [batch_size, 2]
        binary_targets: torch.Tensor,       # [batch_size]
        type_logits: torch.Tensor,          # [batch_size, num_types]
        type_targets: torch.Tensor,         # [batch_size]
        type_mask: Optional[torch.Tensor] = None,  # [batch_size] 表示 type_label 是否有效
        sample_weights: Optional[torch.Tensor] = None  # [batch_size] 强/弱标注权重
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            binary_logits: 二分类 logits
            binary_targets: 二分类目标标签
            type_logits: 类型分类 logits
            type_targets: 类型分类目标标签
            type_mask: 标记哪些样本有有效的 type_label
                       若为 None，则所有样本都被认为有有效 type_label
            sample_weights: 样本权重（strong=1.0, weak=0.4）
        
        Returns:
            dict 包含：
                - 'total_loss': 总损失
                - 'binary_loss': 二分类损失
                - 'type_loss': 类型分类损失
                - 'loss_dict': 详细损失字典
        """
        batch_size = binary_logits.shape[0]
        device = binary_logits.device
        
        # 计算二分类损失
        binary_loss = self.binary_loss_fn(
            binary_logits, binary_targets, 
            sample_weights=sample_weights
        )
        binary_loss = binary_loss.mean()  # 标量
        
        # 计算类型分类损失（仅在有有效 type_label 的样本上）
        if type_mask is None:
            # 默认所有样本都有有效 type_label
            type_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        type_mask = type_mask.to(device)
        
        if type_mask.sum() > 0:
            # 只在 type_mask=True 的样本上计算 type loss
            type_logits_masked = type_logits[type_mask]
            type_targets_masked = type_targets[type_mask]
            sample_weights_masked = (
                sample_weights[type_mask] 
                if sample_weights is not None 
                else None
            )
            
            type_loss = self.type_loss_fn(
                type_logits_masked, type_targets_masked,
                sample_weights=sample_weights_masked
            )
            type_loss = type_loss.mean()
        else:
            # 没有有效 type_label 样本，type_loss = 0
            type_loss = torch.tensor(0.0, device=device)
        
        # 多任务损失合并
        total_loss = (
            self.binary_loss_weight * binary_loss +
            self.type_loss_weight * type_loss
        )
        
        return {
            'total_loss': total_loss,
            'binary_loss': binary_loss.item(),
            'type_loss': type_loss.item() if isinstance(type_loss, torch.Tensor) else type_loss,
            'loss_dict': {
                'total': total_loss.item(),
                'binary': binary_loss.item(),
                'type': type_loss.item() if isinstance(type_loss, torch.Tensor) else type_loss
            }
        }


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    根据标签分布自动计算类别权重
    
    策略：类权重 = 1 / (类样本数)，再归一化
    
    Args:
        labels: 标签数组，形状 [batch_size]
        num_classes: 类别数
        device: 计算设备
    
    Returns:
        weights: 类权重，形状 [num_classes]
    """
    # 统计各类样本数
    class_counts = torch.bincount(labels, minlength=num_classes)
    
    # 防止除零
    class_counts = class_counts.float().to(device)
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # 计算权重：样本少的类权重高
    weights = 1.0 / class_counts
    
    # 归一化：使得平均权重为 1
    weights = weights / weights.sum() * num_classes
    
    return weights


def compute_sample_weights_from_label_type(
    label_types: list,  # ["strong", "weak", ...]
    strong_weight: float = 1.0,
    weak_weight: float = 0.4,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    根据 label_type（强/弱标注）计算样本权重
    
    Args:
        label_types: 每个样本的标注类型列表 ["strong", "weak", ...]
        strong_weight: 强标注样本的权重（默认 1.0）
        weak_weight: 弱标注样本的权重（默认 0.4）
        device: 计算设备
    
    Returns:
        weights: 样本权重，形状 [batch_size]
    """
    weights = []
    for label_type in label_types:
        if label_type == "weak":
            weights.append(weak_weight)
        elif label_type == "strong" or label_type is None:
            weights.append(strong_weight)
        else:
            weights.append(strong_weight)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_type_mask_from_labels(
    type_labels: torch.Tensor,
    invalid_type_id: int = -1,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    根据 type_label 计算 type_mask（判断是否有有效的 type_label）
    
    Args:
        type_labels: 类型标签，形状 [batch_size]
        invalid_type_id: 无效 type_id 的值（默认 -1）
        device: 计算设备
    
    Returns:
        mask: 布尔掩码，True 表示有有效 type_label，形状 [batch_size]
    """
    mask = (type_labels != invalid_type_id).to(device)
    return mask


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试损失函数模块")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 测试数据
    batch_size = 8
    num_classes_binary = 2
    num_classes_type = 6
    
    # 模拟 logits 和 targets
    binary_logits = torch.randn(batch_size, num_classes_binary, device=device)
    binary_targets = torch.randint(0, 2, (batch_size,), device=device)
    
    type_logits = torch.randn(batch_size, num_classes_type, device=device)
    type_targets = torch.randint(0, num_classes_type, (batch_size,), device=device)
    
    # 模拟 type_mask 和 sample_weights
    type_mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=device)
    sample_weights = torch.ones(batch_size, device=device)
    sample_weights[::2] = 0.4  # 每隔一个样本权重为 0.4（弱标注）
    
    print("✅ 测试数据创建成功")
    print(f"   binary_logits shape: {binary_logits.shape}")
    print(f"   type_logits shape: {type_logits.shape}")
    
    # 测试 FocalLoss
    print("\n--- FocalLoss ---")
    focal_loss = FocalLoss(gamma=2.5)
    loss = focal_loss(binary_logits, binary_targets, sample_weights=sample_weights)
    print(f"✅ FocalLoss: {loss.item():.6f}")
    
    # 测试 WeightedCrossEntropyLoss
    print("\n--- WeightedCrossEntropyLoss ---")
    class_weights = compute_class_weights(binary_targets, num_classes_binary, device)
    print(f"计算的类权重: {class_weights}")
    wce_loss = WeightedCrossEntropyLoss(class_weights=class_weights)
    loss = wce_loss(binary_logits, binary_targets, sample_weights=sample_weights)
    print(f"✅ WeightedCrossEntropyLoss: {loss.item():.6f}")
    
    # 测试 MultiTaskLoss
    print("\n--- MultiTaskLoss ---")
    binary_class_weights = compute_class_weights(binary_targets, num_classes_binary, device)
    type_class_weights = compute_class_weights(type_targets, num_classes_type, device)
    
    multi_loss = MultiTaskLoss(
        binary_loss_weight=1.0,
        type_loss_weight=1.0,
        use_focal_loss=True,
        focal_gamma=2.5,
        binary_class_weights=binary_class_weights,
        type_class_weights=type_class_weights
    )
    
    loss_dict = multi_loss(
        binary_logits, binary_targets,
        type_logits, type_targets,
        type_mask=type_mask,
        sample_weights=sample_weights
    )
    
    print(f"✅ MultiTaskLoss 计算成功")
    print(f"   总损失: {loss_dict['total_loss'].item():.6f}")
    print(f"   二分类损失: {loss_dict['binary_loss']:.6f}")
    print(f"   类型分类损失: {loss_dict['type_loss']:.6f}")
    
    # 测试 compute_sample_weights_from_label_type
    print("\n--- compute_sample_weights_from_label_type ---")
    label_types = ["strong", "weak", "strong", "weak"] * 2
    sample_weights = compute_sample_weights_from_label_type(
        label_types, device=device
    )
    print(f"✅ 样本权重: {sample_weights}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
