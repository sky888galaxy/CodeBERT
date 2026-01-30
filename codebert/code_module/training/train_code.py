"""
CodeBERT 恶意代码检测器 - 主训练脚本

功能：
1. 多任务学习训练（二分类 + 类型分类）
2. FocalLoss 处理类不平衡
3. 弱/强标注样本差异化权重
4. 梯度累积、warmup、scheduler
5. Early stopping
6. 完整的指标记录与模型保存
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TrainingConfig, DEVICE, MODEL_SAVE_DIR, RESULTS_DIR, TYPE_LABELS
from models.codebert_detector import CodeBertDetector
from training.dataset import create_data_loaders, load_data_from_json, analyze_dataset_statistics, print_dataset_statistics
from training.losses import (
    MultiTaskLoss, compute_class_weights, compute_sample_weights_from_label_type,
    compute_type_mask_from_labels
)
from training.metrics import MetricsComputer


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 3, metric: str = 'val_f1', mode: str = 'max'):
        """
        Args:
            patience: 等待轮数（多少个 epoch 无提升后停止）
            metric: 监控指标（'val_f1', 'val_loss', 'val_roc_auc'）
            mode: 'max' (指标越大越好) 或 'min' (指标越小越好)
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        
        self.best_score = None
        self.patience_counter = 0
        self.stop = False
    
    def check(self, current_score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_score: 当前指标值
        
        Returns:
            True 表示应该停止训练
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        # 判断是否有改进
        if self.mode == 'max':
            improved = current_score > self.best_score
        else:
            improved = current_score < self.best_score
        
        if improved:
            self.best_score = current_score
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            return False


class CodeBertTrainer:
    """训练器"""
    
    def __init__(
        self,
        model: CodeBertDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        config: TrainingConfig = TrainingConfig(),
        device: torch.device = DEVICE
    ):
        """
        Args:
            model: CodeBERTDetector 模型
            train_loader: 训练 DataLoader
            val_loader: 验证 DataLoader
            test_loader: 测试 DataLoader（可选）
            config: 训练配置
            device: 计算设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度
        total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 损失函数
        self._init_loss_fn()
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            metric=config.early_stopping_metric,
            mode='max' if 'f1' in config.early_stopping_metric or 'auc' in config.early_stopping_metric else 'min'
        )
        
        # 历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # 其他
        self.global_step = 0
        self.best_val_metric = None
        self.best_model_path = None
    
    def _init_loss_fn(self):
        """初始化损失函数"""
        # 计算类权重
        binary_labels = []
        type_labels = []
        
        for batch in self.train_loader:
            binary_labels.extend(batch['binary_label'].cpu().numpy())
            # 仅统计有有效类型标签的样本，避免 -1 污染类别权重
            mask_np = batch['type_mask'].cpu().numpy().astype(bool)
            if mask_np.any():
                type_labels.extend(batch['type_label'].cpu().numpy()[mask_np])
        
        binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        type_labels = torch.tensor(type_labels, dtype=torch.long)
        
        binary_class_weights = compute_class_weights(
            binary_labels, num_classes=2, device=self.device
        )
        if len(type_labels) > 0:
            type_class_weights = compute_class_weights(
                torch.tensor(type_labels, dtype=torch.long, device=self.device),
                num_classes=len(TYPE_LABELS), device=self.device
            )
        else:
            # 若全为无效类型，退化为均匀权重
            type_class_weights = torch.ones(len(TYPE_LABELS), device=self.device)
        
        print(f"\n📊 自动计算的类权重:")
        print(f"   二分类: {binary_class_weights.tolist()}")
        print(f"   类型分类: {type_class_weights.tolist()}")
        
        self.loss_fn = MultiTaskLoss(
            binary_loss_weight=self.config.binary_loss_weight,
            type_loss_weight=self.config.type_loss_weight,
            use_focal_loss=self.config.use_focal_loss,
            focal_gamma=2.5,
            binary_class_weights=binary_class_weights,
            type_class_weights=type_class_weights
        )
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个 epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs} [TRAIN]")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            binary_label = batch['binary_label'].to(self.device)
            type_label = batch['type_label'].to(self.device)
            type_mask = batch['type_mask'].to(self.device)
            sample_weight = batch['sample_weight'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            binary_logits = outputs['binary_logits']
            type_logits = outputs['type_logits']
            
            # 计算损失
            loss_dict = self.loss_fn(
                binary_logits=binary_logits,
                binary_targets=binary_label,
                type_logits=type_logits,
                type_targets=type_label,
                type_mask=type_mask,
                sample_weights=sample_weight
            )
            
            loss = loss_dict['total_loss']
            
            # 梯度累积
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # 梯度累积步
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 日志
            if (batch_idx + 1) % self.config.log_every_n_steps == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss / num_batches:.4f}',
                    'binary_loss': f'{loss_dict["binary_loss"]:.4f}',
                    'type_loss': f'{loss_dict["type_loss"]:.4f}'
                })
        
        # 如果最后不足一个 accumulation 步的梯度仍在累积，补一次 step
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Tuple[float, Dict]:
        """
        验证一个 epoch
        
        Returns:
            (平均验证损失, 验证指标字典)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_binary_preds = []
        all_binary_labels = []
        all_binary_probs = []
        all_type_preds = []
        all_type_labels = []
        all_type_probs = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs} [VAL]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            binary_label = batch['binary_label'].to(self.device)
            type_label = batch['type_label'].to(self.device)
            type_mask = batch['type_mask'].to(self.device)
            sample_weight = batch['sample_weight'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            binary_logits = outputs['binary_logits']
            type_logits = outputs['type_logits']
            
            # 计算损失
            loss_dict = self.loss_fn(
                binary_logits=binary_logits,
                binary_targets=binary_label,
                type_logits=type_logits,
                type_targets=type_label,
                type_mask=type_mask,
                sample_weights=sample_weight
            )
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
            
            # 收集预测结果
            binary_probs = torch.softmax(binary_logits, dim=1)
            binary_pred = torch.argmax(binary_logits, dim=1)
            
            type_probs = torch.softmax(type_logits, dim=1)
            type_pred = torch.argmax(type_logits, dim=1)
            
            all_binary_preds.extend(binary_pred.cpu().numpy())
            all_binary_labels.extend(binary_label.cpu().numpy())
            all_binary_probs.extend(binary_probs.cpu().numpy())
            
            # 仅在 type_mask=True 的样本上收集类型指标，避免无类型样本污染
            type_mask_np = type_mask.cpu().numpy().astype(bool)
            if type_mask_np.any():
                all_type_preds.extend(type_pred.cpu().numpy()[type_mask_np])
                all_type_labels.extend(type_label.cpu().numpy()[type_mask_np])
                all_type_probs.extend(type_probs.cpu().numpy()[type_mask_np])
        
        # 计算指标
        binary_preds = np.array(all_binary_preds)
        binary_labels = np.array(all_binary_labels)
        binary_probs = np.array(all_binary_probs)
        
        type_preds = np.array(all_type_preds)
        type_labels = np.array(all_type_labels)
        type_probs = np.array(all_type_probs)
        
        # 二分类指标
        binary_computer = MetricsComputer(num_classes=2, task_name="binary")
        binary_metrics = binary_computer.compute_metrics(
            binary_labels, binary_preds, binary_probs
        )
        
        # 类型分类指标
        if len(all_type_labels) > 0:
            type_computer = MetricsComputer(num_classes=len(TYPE_LABELS), task_name="multiclass")
            type_metrics = type_computer.compute_metrics(
                type_labels, type_preds, type_probs
            )
        else:
            # 无有效类型标签，跳过类型指标
            type_metrics = {'note': 'no valid type labels in validation set'}
        
        avg_val_loss = total_loss / num_batches
        
        val_metrics = {
            'val_loss': avg_val_loss,
            'val_binary_metrics': binary_metrics,
            'val_type_metrics': type_metrics
        }
        
        self.history['val_loss'].append(avg_val_loss)
        self.history['val_metrics'].append(val_metrics)
        
        return avg_val_loss, val_metrics
    
    def train(self):
        """完整训练流程"""
        print("=" * 80)
        print("开始训练 CodeBERT 恶意代码检测器")
        print("=" * 80)
        print(f"设备: {self.device}")
        print(f"最大 epochs: {self.config.max_epochs}")
        print(f"批大小: {self.config.batch_size}")
        print(f"学习率: {self.config.learning_rate}")
        print(f"使用 FocalLoss: {self.config.use_focal_loss}")
        print("=" * 80 + "\n")
        
        for epoch in range(self.config.max_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"\n✅ Epoch {epoch+1} 训练损失: {train_loss:.6f}")
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            print(f"✅ Epoch {epoch+1} 验证损失: {val_loss:.6f}")
            
            # 打印二分类指标
            binary_metrics = val_metrics['val_binary_metrics']
            print(f"   二分类 F1: {binary_metrics.get('f1', 0):.4f}")
            print(f"   二分类 ROC-AUC: {binary_metrics.get('roc_auc', 0):.4f}")
            
            # 早停检查
            monitor_metric_value = binary_metrics.get('f1', 0)
            
            # 保存最佳模型
            if self.best_val_metric is None or monitor_metric_value > self.best_val_metric:
                self.best_val_metric = monitor_metric_value
                self._save_best_model(epoch)
                print(f"   🎯 保存最佳模型（F1={monitor_metric_value:.4f}）")
            
            # 检查早停
            if self.early_stopping.check(monitor_metric_value):
                print(f"\n⏹️  触发早停（patience={self.config.early_stopping_patience}）")
                break
        
        print("\n" + "=" * 80)
        print("✅ 训练完成！")
        print("=" * 80)
        
        # 保存训练历史
        self._save_history()
    
    def _save_best_model(self, epoch: int):
        """保存最佳模型"""
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = MODEL_SAVE_DIR / f"best_model_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 encoder 权重与完整 state_dict，避免访问不存在的 self.model.model
        self.model.encoder.save_pretrained(model_dir / "encoder")
        # 同步保存 tokenizer，确保推理与训练一致
        try:
            tokenizer = self.train_loader.dataset.tokenizer  # 训练使用的 tokenizer
            tokenizer.save_pretrained(model_dir / "tokenizer")
        except Exception:
            pass
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        
        # 保存配置
        config_dict = {
            'model_name': self.model.model_name,
            'hidden_size': self.model.hidden_size,
            'num_types': self.model.num_types,
            'epoch': epoch,
            'best_metric': self.best_val_metric
        }
        
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.best_model_path = model_dir
        print(f"   模型已保存到: {model_dir}")
    
    def _save_history(self):
        """保存训练历史"""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = RESULTS_DIR / f"training_history_{timestamp}.json"
        
        # 转换为可序列化格式
        serializable_history = {
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'best_val_metric': self.best_val_metric
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"✅ 训练历史已保存到: {history_path}")


# ==================== 主程序 ====================
def main():
    """主训练函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeBERT 恶意代码检测器训练脚本")
    parser.add_argument(
        "--train-data", "-td",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "train.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--val-data", "-vd",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "val.json",
        help="验证数据路径"
    )
    parser.add_argument(
        "--test-data", "-ted",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "test.json",
        help="测试数据路径"
    )
    parser.add_argument(
        "--batch-size", "-bs",
        type=int,
        default=TrainingConfig.batch_size,
        help="批大小"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=TrainingConfig.max_epochs,
        help="最大轮数"
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print("⏳ 加载数据...")
    train_data = load_data_from_json(args.train_data)
    val_data = load_data_from_json(args.val_data)
    test_data = load_data_from_json(args.test_data) if args.test_data.exists() else None
    
    # 打印数据统计
    print(f"\n📊 训练集统计:")
    stats = analyze_dataset_statistics(train_data)
    print_dataset_statistics(stats)
    
    # 创建 DataLoaders
    loaders = create_data_loaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size
    )
    
    # 创建模型
    print("⏳ 创建模型...")
    model = CodeBertDetector()
    print(f"✅ 模型创建成功")
    print(f"   参数量: {model.get_num_params()}")
    
    # 更新配置
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.max_epochs = args.epochs
    
    # 创建训练器
    trainer = CodeBertTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        test_loader=loaders.get('test'),
        config=config,
        device=DEVICE
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
