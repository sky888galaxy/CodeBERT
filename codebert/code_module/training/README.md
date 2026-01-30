# 训练模块说明

## 概述

本模块提供完整的模型训练流水线，包括：
- 数据加载与预处理
- 损失函数定义（FocalLoss + MultiTask）
- 评估指标计算
- 训练循环与优化
- 模型保存与日志记录

## 文件结构

```
training/
├── README.md                    # 本文件
├── dataset.py                   # 数据集定义与加载
├── losses.py                    # 损失函数（FocalLoss, MultiTask）
├── metrics.py                   # 评估指标计算
└── train_code.py                # 主训练脚本
```

## 核心模块说明

### 1. dataset.py - 数据加载与预处理

#### 主要类：`MaliciousCodeDataset`

```python
from training.dataset import MaliciousCodeDataset, create_data_loaders

# 加载数据
train_data = load_data_from_json("train.json")

# 创建 DataLoader
loaders = create_data_loaders(
    train_data, val_data, test_data,
    batch_size=16,
    model_name="microsoft/codebert-base"
)

# 使用 DataLoader
for batch in loaders['train']:
    input_ids = batch['input_ids']           # [batch_size, seq_len]
    binary_label = batch['binary_label']     # [batch_size]
    type_label = batch['type_label']         # [batch_size]
    type_mask = batch['type_mask']           # [batch_size] 是否有有效 type_label
    sample_weight = batch['sample_weight']   # [batch_size] 强/弱标注权重
```

**样本格式：**
```python
{
    'input_ids': torch.Tensor,               # 分词后的 token IDs
    'attention_mask': torch.Tensor,          # 注意力掩码
    'binary_label': torch.Tensor,            # 二分类标签 (0/1)
    'type_label': torch.Tensor,              # 恶意类型标签 (0-5)
    'label_type': str,                       # "strong" 或 "weak"
    'type_mask': torch.Tensor,               # type_label 是否有效
    'sample_weight': torch.Tensor            # 样本权重 (1.0 or 0.4)
}
```

#### 关键函数：

| 函数 | 功能 |
|---|---|
| `load_data_from_json()` | 从 JSON 文件加载数据 |
| `create_data_loaders()` | 创建训练/验证/测试 DataLoader |
| `analyze_dataset_statistics()` | 分析数据集统计信息 |
| `print_dataset_statistics()` | 打印可视化统计 |

### 2. losses.py - 损失函数

#### FocalLoss

处理类别严重不平衡，重新加权简单/困难样本。

```python
from training.losses import FocalLoss

focal_loss = FocalLoss(
    alpha=torch.tensor([0.25, 0.75]),  # 类别权重
    gamma=2.5,                          # 难度系数
    reduction='mean'
)

loss = focal_loss(logits, targets, sample_weights)
```

**公式：**
$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$

其中 $\gamma=0$ 时退化为加权 CrossEntropy，$\gamma>0$ 时对困难样本加权。

#### MultiTaskLoss

合并二分类和类型分类损失，支持 type_label 无效样本的 mask。

```python
from training.losses import MultiTaskLoss

multi_loss = MultiTaskLoss(
    binary_loss_weight=1.0,
    type_loss_weight=1.0,
    use_focal_loss=True,
    focal_gamma=2.5,
    binary_class_weights=binary_weights,
    type_class_weights=type_weights
)

loss_dict = multi_loss(
    binary_logits=binary_logits,
    binary_targets=binary_targets,
    type_logits=type_logits,
    type_targets=type_targets,
    type_mask=type_mask,                  # 标记哪些样本有有效 type_label
    sample_weights=sample_weights         # 强/弱标注权重
)

print(loss_dict['total_loss'])    # 总损失
print(loss_dict['binary_loss'])   # 二分类损失
print(loss_dict['type_loss'])     # 类型分类损失
```

**多任务损失公式：**
$$L = L_{binary} + \lambda \cdot L_{type}$$

#### 自适应权重计算

```python
from training.losses import compute_class_weights, compute_sample_weights_from_label_type

# 计算类别权重（根据样本分布）
class_weights = compute_class_weights(labels, num_classes=2, device=device)

# 计算样本权重（强/弱标注）
sample_weights = compute_sample_weights_from_label_type(
    label_types=['strong', 'weak', ...],
    strong_weight=1.0,
    weak_weight=0.4,
    device=device
)
```

### 3. metrics.py - 评估指标

#### MetricsComputer 类

```python
from training.metrics import MetricsComputer

computer = MetricsComputer(num_classes=2, task_name="binary")

metrics = computer.compute_metrics(
    y_true=np.array([0, 1, 1, 0, ...]),
    y_pred=np.array([0, 1, 1, 0, ...]),
    y_prob=np.array([[0.9, 0.1], [0.2, 0.8], ...])
)

# 打印指标
computer.print_metrics(metrics)

# 保存指标
computer.save_metrics(metrics, "metrics.json")
```

#### 支持的指标

**二分类：**
- Accuracy（准确率）
- Precision / Recall / F1（精密/召回/F1）
- ROC-AUC（接收者操作特征曲线下面积）
- PR-AUC（精确率-召回率曲线下面积）
- MCC（Matthews Correlation Coefficient）
- Confusion Matrix（混淆矩阵）

**多分类：**
- 上述所有指标的 Macro 和 Micro 平均
- Per-class 指标

### 4. train_code.py - 主训练脚本

#### 快速开始

```bash
python train_code.py \
  --train-data ../data/train.json \
  --val-data ../data/val.json \
  --test-data ../data/test.json \
  --batch-size 16 \
  --epochs 15
```

#### 工作流

1. **数据加载与统计**
   ```
   ⏳ 加载数据...
   ✅ 从 train.json 加载 12000 条数据
   📊 训练集统计:
      总样本数: 12000
      正常代码: 4500 (37.5%)
      恶意代码: 7500 (62.5%)
   ```

2. **模型初始化**
   ```
   ⏳ 创建模型...
   ✅ 模型创建成功
      参数量: {'total': 108M, 'trainable': 108M}
   ```

3. **损失函数初始化**
   ```
   📊 自动计算的类权重:
      二分类: [1.67, 0.89]     # 恶意样本权重更高
      类型分类: [2.5, 1.2, ...]
   ```

4. **训练循环**
   ```
   Epoch 1/15 [TRAIN]: 100%|████| 750/750 [5:30<00:00, 2.26 batch/s]
   ✅ Epoch 1 训练损失: 0.450123
   Epoch 1/15 [VAL]: 100%|████| 94/94 [0:45<00:00, 0.48 batch/s]
   ✅ Epoch 1 验证损失: 0.380456
      二分类 F1: 0.8245
      二分类 ROC-AUC: 0.8934
      🎯 保存最佳模型（F1=0.8245）
   ```

5. **模型保存**
   ```
   ✅ 模型已保存到: ../experiments/runs/best_model_20251207_120530
      - encoder/         # HuggingFace 格式的预训练模型
      - model.pt         # 完整模型权重
      - config.json      # 配置文件
   ```

#### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `batch_size` | 16 | 批大小 |
| `learning_rate` | 5e-5 | 学习率 |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `max_epochs` | 15 | 最大轮数 |
| `warmup_ratio` | 0.1 | Warmup 步数占比 |
| `gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `early_stopping_patience` | 3 | Early Stopping 耐心（轮数） |
| `use_focal_loss` | True | 是否使用 Focal Loss |

#### 早停机制

训练会自动监控验证集 F1 分数，如果连续 3 个 epoch 无改进，则停止训练。

## 训练最佳实践

### 1. 处理类不平衡

系统会自动根据训练集计算类权重，无需手动调整：

```python
# 自动计算的权重（更多样本的类权重更低）
class_weight[0] = n_total / (2 * n_class_0)
class_weight[1] = n_total / (2 * n_class_1)
```

### 2. 弱标注样本处理

弱标注样本的损失会自动降权：

```python
if label_type == "strong":
    sample_weight = 1.0
else:  # weak
    sample_weight = 0.4
```

可在 `config.py` 中调整：
```python
strong_sample_weight = 1.0
weak_sample_weight = 0.4
```

### 3. 梯度累积

如果显存有限，可增加梯度累积步数以达到更大的有效 batch size：

```python
# 在 config.py 中
gradient_accumulation_steps = 2  # 有效 batch_size = 16 * 2 = 32
```

### 4. 学习率调度

默认使用线性 warmup + 线性衰减的 scheduler：

```
Learning Rate
     |     ___________
     |    /           \___
     |   /                 \___
     |  /
     |_/________________________ Steps
        warmup            max_steps
```

### 5. 多 GPU 训练（可选扩展）

如需多 GPU 分布式训练，可在 `train_code.py` 中添加 `DistributedDataParallel`：

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 常见问题

### Q1: 训练时 GPU 内存不足

答：尝试以下方法：
1. 减小 `batch_size`（默认 16，可降至 8 或 4）
2. 增加 `gradient_accumulation_steps`（保持有效 batch size）
3. 冻结 encoder 部分参数（见 `models/codebert_detector.py`）

### Q2: 验证损失不下降

答：
1. 检查学习率是否过高/过低（建议 2e-5 ~ 1e-4）
2. 增加 warmup 步数（`warmup_ratio` 增加至 0.2）
3. 检查数据质量（数据泄露、标注错误）
4. 尝试 FocalLoss（自动处理类不平衡）

### Q3: 如何加载训练好的模型进行微调？

答：
```python
from training.train_code import CodeBertTrainer
from models.codebert_detector import CodeBertDetector

# 加载已保存的模型
model = CodeBertDetector()
model.load_state_dict(torch.load("path/to/model.pt"))

# 继续训练
trainer = CodeBertTrainer(model, train_loader, val_loader, ...)
trainer.train()
```

## 输出目录结构

```
experiments/
├── runs/
│   ├── best_model_20251207_120530/
│   │   ├── encoder/
│   │   │   ├── config.json
│   │   │   ├── pytorch_model.bin
│   │   │   └── ...
│   │   ├── model.pt              # 完整模型权重
│   │   └── config.json           # 训练配置
│   └── ...
└── results/
    ├── training_history_20251207_120530.json
    ├── val_metrics_epoch_1.json
    └── ...
```

## 下一步

- 查看 `../inference/README.md` 了解如何使用训练好的模型进行推理
- 查看 `../data/README.md` 了解数据处理和主动学习集成
- 查看 `../models/README.md` 了解模型架构详解
