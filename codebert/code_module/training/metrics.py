"""
完整的评估指标模块

包含：
1. 基础指标：Accuracy, Precision, Recall, F1
2. 高级指标：ROC-AUC, PR-AUC, MCC
3. 混淆矩阵
4. 每类指标计算
5. JSON 输出友好格式
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    confusion_matrix, roc_curve, classification_report
)
import json
from pathlib import Path


class MetricsComputer:
    """评估指标计算器"""
    
    def __init__(self, num_classes: int = 2, task_name: str = "binary"):
        """
        Args:
            num_classes: 类别数（二分类=2，多分类>2）
            task_name: 任务名称 ("binary" 或 "multiclass")
        """
        self.num_classes = num_classes
        self.task_name = task_name
        self.num_samples = 0
    
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签，形状 [n_samples]
            y_pred: 预测标签，形状 [n_samples]
            y_prob: 预测概率，形状 [n_samples, num_classes] 或 [n_samples]
        
        Returns:
            metrics_dict: 包含所有指标的字典
        """
        # 转换为 numpy
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        y_prob = self._to_numpy(y_prob)
        
        self.num_samples = len(y_true)
        
        metrics = {}
        
        # 1. 基础指标
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # 2. 精度、召回、F1（按 macro 和 micro）
        if self.num_classes == 2:
            # 二分类：使用 "binary" 方式，更符合通常含义
            metrics['precision'] = float(precision_score(
                y_true, y_pred, average='binary', zero_division=0
            ))
            metrics['recall'] = float(recall_score(
                y_true, y_pred, average='binary', zero_division=0
            ))
            metrics['f1'] = float(f1_score(
                y_true, y_pred, average='binary', zero_division=0
            ))
            metrics['precision_macro'] = float(precision_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
            metrics['recall_macro'] = float(recall_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
            metrics['f1_macro'] = float(f1_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
        else:
            # 多分类：使用 macro 和 micro
            metrics['precision_macro'] = float(precision_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
            metrics['recall_macro'] = float(recall_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
            metrics['f1_macro'] = float(f1_score(
                y_true, y_pred, average='macro', zero_division=0
            ))
            metrics['precision_micro'] = float(precision_score(
                y_true, y_pred, average='micro', zero_division=0
            ))
            metrics['recall_micro'] = float(recall_score(
                y_true, y_pred, average='micro', zero_division=0
            ))
            metrics['f1_micro'] = float(f1_score(
                y_true, y_pred, average='micro', zero_division=0
            ))
        
        # 3. AUC（仅二分类）
        if self.num_classes == 2:
            try:
                # ROC-AUC
                if y_prob.ndim == 2:
                    # 形状 [n_samples, 2]，使用正类概率
                    y_prob_positive = y_prob[:, 1]
                else:
                    # 形状 [n_samples]，直接使用
                    y_prob_positive = y_prob
                
                roc_auc = roc_auc_score(y_true, y_prob_positive)
                metrics['roc_auc'] = float(roc_auc)
                
                # PR-AUC
                precision_vals, recall_vals, _ = precision_recall_curve(
                    y_true, y_prob_positive
                )
                pr_auc = auc(recall_vals, precision_vals)
                metrics['pr_auc'] = float(pr_auc)
            except Exception as e:
                print(f"⚠️  AUC 计算失败: {e}")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        else:
            # 多分类 AUC（OvR）
            try:
                if y_prob.ndim == 2:
                    roc_auc_ovr = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )
                    metrics['roc_auc_macro'] = float(roc_auc_ovr)
            except Exception as e:
                print(f"⚠️  多分类 AUC 计算失败: {e}")
                metrics['roc_auc_macro'] = None
        
        # 4. Matthews Correlation Coefficient (MCC)
        if self.num_classes == 2:
            try:
                mcc = matthews_corrcoef(y_true, y_pred)
                metrics['mcc'] = float(mcc)
            except Exception as e:
                print(f"⚠️  MCC 计算失败: {e}")
                metrics['mcc'] = None
        
        # 5. 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        metrics['confusion_matrix'] = cm.tolist()
        
        # 6. 每类指标
        metrics['per_class'] = self._compute_per_class_metrics(y_true, y_pred)
        
        return metrics
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """计算每类的精度、召回、F1"""
        per_class = {}
        
        # 使用 classification_report 获得详细信息
        report_dict = classification_report(
            y_true, y_pred, 
            labels=list(range(self.num_classes)),
            output_dict=True,
            zero_division=0
        )
        
        for class_id in range(self.num_classes):
            class_key = str(class_id)
            if class_key in report_dict:
                per_class[class_key] = {
                    'precision': float(report_dict[class_key]['precision']),
                    'recall': float(report_dict[class_key]['recall']),
                    'f1': float(report_dict[class_key]['f1-score']),
                    'support': int(report_dict[class_key]['support'])
                }
        
        return per_class
    
    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """将 tensor 转换为 numpy 数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    
    def print_metrics(self, metrics: Dict) -> None:
        """打印指标（格式化）"""
        print("\n" + "=" * 60)
        print(f"评估指标 ({self.task_name})")
        print("=" * 60)
        
        # 基础指标
        print(f"准确率 (Accuracy):        {metrics.get('accuracy', 0):.4f}")
        
        if self.num_classes == 2:
            print(f"精确率 (Precision):       {metrics.get('precision', 0):.4f}")
            print(f"召回率 (Recall):          {metrics.get('recall', 0):.4f}")
            print(f"F1 分数:                 {metrics.get('f1', 0):.4f}")
            print(f"F1 分数 (Macro):         {metrics.get('f1_macro', 0):.4f}")
            print(f"ROC-AUC:                {metrics.get('roc_auc', 0):.4f}")
            print(f"PR-AUC:                 {metrics.get('pr_auc', 0):.4f}")
            print(f"MCC:                    {metrics.get('mcc', 0):.4f}")
        else:
            print(f"精确率 (Macro):          {metrics.get('precision_macro', 0):.4f}")
            print(f"召回率 (Macro):          {metrics.get('recall_macro', 0):.4f}")
            print(f"F1 分数 (Macro):         {metrics.get('f1_macro', 0):.4f}")
            print(f"精确率 (Micro):          {metrics.get('precision_micro', 0):.4f}")
            print(f"召回率 (Micro):          {metrics.get('recall_micro', 0):.4f}")
            print(f"F1 分数 (Micro):         {metrics.get('f1_micro', 0):.4f}")
            if metrics.get('roc_auc_macro'):
                print(f"ROC-AUC (Macro):        {metrics.get('roc_auc_macro', 0):.4f}")
        
        # 混淆矩阵
        print(f"\n混淆矩阵:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
        
        print("=" * 60 + "\n")
    
    def save_metrics(self, metrics: Dict, save_path: Union[str, Path]) -> None:
        """保存指标到 JSON 文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理 numpy 数组（转为列表）
        metrics_serializable = self._make_serializable(metrics)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 指标已保存到: {save_path}")
    
    @staticmethod
    def _make_serializable(obj):
        """递归地将 numpy/torch 对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: MetricsComputer._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [MetricsComputer._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算二分类指标（简化版）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选，用于 AUC）
    
    Returns:
        指标字典
    """
    computer = MetricsComputer(num_classes=2, task_name="binary")
    if y_prob is None:
        y_prob = np.eye(2)[y_pred]  # 从 pred 构造 one-hot
    return computer.compute_metrics(y_true, y_pred, y_prob)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 6
) -> Dict[str, float]:
    """
    计算多分类指标（简化版）
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        num_classes: 类别数
    
    Returns:
        指标字典
    """
    computer = MetricsComputer(num_classes=num_classes, task_name="multiclass")
    if y_prob is None:
        y_prob = np.eye(num_classes)[y_pred]
    return computer.compute_metrics(y_true, y_pred, y_prob)


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("测试评估指标模块")
    print("=" * 60)
    
    # 测试二分类
    print("\n--- 二分类 ---")
    y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_binary = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    y_prob_binary = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.8, 0.2],
        [0.6, 0.4],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.2, 0.8]
    ])
    
    computer_binary = MetricsComputer(num_classes=2, task_name="binary")
    metrics_binary = computer_binary.compute_metrics(y_true_binary, y_pred_binary, y_prob_binary)
    computer_binary.print_metrics(metrics_binary)
    
    # 测试多分类
    print("\n--- 多分类 ---")
    y_true_multi = np.array([0, 1, 2, 1, 2, 0, 1, 2, 1, 0])
    y_pred_multi = np.array([0, 1, 2, 1, 1, 0, 2, 2, 1, 0])
    y_prob_multi = np.random.dirichlet([1, 1, 1], size=10)
    
    computer_multi = MetricsComputer(num_classes=3, task_name="multiclass")
    metrics_multi = computer_multi.compute_metrics(y_true_multi, y_pred_multi, y_prob_multi)
    computer_multi.print_metrics(metrics_multi)
    
    # 测试保存
    print("\n--- 保存指标 ---")
    computer_binary.save_metrics(metrics_binary, "./test_metrics_binary.json")
    computer_multi.save_metrics(metrics_multi, "./test_metrics_multi.json")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
