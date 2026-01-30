# 模型架构说明

## 概述

本模块定义了 CodeBERT 多任务检测器的架构，支持：
- 多任务学习（二分类 + 多分类）
- 隐藏层提取与可视化
- 灵活的 encoder 替换
- 梯度反向与权重共享

## 文件结构

```
models/
├── __init__.py
├── README.md                    # 本文件
└── codebert_detector.py         # 模型定义
```

## 模型架构

### 整体设计

```
输入代码 (str)
  ↓
Tokenization & Encoding
  ↓
┌─────────────────────┐
│  CodeBERT Encoder   │ (microsoft/codebert-base)
│  - 12 layers        │ 
│  - 12 heads         │
│  - 768 hidden       │
└─────────────────────┘
  ↓
[CLS] Token Pooling
  ↓
Dropout (p=0.1)
  ↓
┌──────────────────────┐
│ Binary Classification│ ──→ P(Safe/Malicious)
│ Head: Linear(768→2)  │
└──────────────────────┘
  ↓
┌──────────────────────┐
│ Type Classification  │ ──→ P(Type_0...Type_5)
│ Head: Linear(768→6)  │
└──────────────────────┘
  ↓
输出：{binary_logits, type_logits, [hidden]}
```

### 数学表示

**编码层：**
$$H = \text{Encoder}(x) \in \mathbb{R}^{L \times 768}$$

其中 $L$ 为序列长度，$x$ 为 token IDs。

**池化层：**
$$h = H[:, 0, :] \in \mathbb{R}^{768}$$

取 [CLS] token 的表示作为句子级特征。

**分类头：**
$$\hat{y}_{\text{binary}} = \text{Softmax}(W_b h + b_b)$$
$$\hat{y}_{\text{type}} = \text{Softmax}(W_t h + b_t)$$

其中 $W_b \in \mathbb{R}^{2 \times 768}$，$W_t \in \mathbb{R}^{6 \times 768}$。

## 核心类：CodeBertDetector

### 初始化

```python
from models.codebert_detector import CodeBertDetector

model = CodeBertDetector(
    model_name="microsoft/codebert-base",  # 预训练模型名
    hidden_size=768,                        # 隐藏层维度
    num_types=6,                            # 恶意类型数
    dropout_prob=0.1,                       # Dropout 概率
    freeze_encoder=False                    # 是否冻结 encoder
)
```

### 主要方法

#### 1. forward() - 前向传播

```python
outputs = model(
    input_ids=input_ids,              # [batch_size, seq_len]
    attention_mask=attention_mask,    # [batch_size, seq_len]
    return_hidden=False               # 是否返回隐藏表示
)

# 返回
{
    "binary_logits": torch.Tensor,    # [batch_size, 2]
    "type_logits": torch.Tensor,      # [batch_size, num_types]
    "hidden": torch.Tensor            # [batch_size, 768] (可选)
}
```

#### 2. predict() - 推理模式

```python
predictions = model.predict(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_probs=True                 # True: 返回概率，False: 返回 logits
)

# 返回
{
    "binary_probs": torch.Tensor,     # [batch_size, 2]
    "type_probs": torch.Tensor,       # [batch_size, num_types]
    "binary_pred": torch.Tensor,      # [batch_size] 预测标签
    "type_pred": torch.Tensor         # [batch_size] 预测类型
}
```

#### 3. freeze_encoder() / unfreeze_encoder()

```python
# 冻结 encoder（只训练分类头）
model.freeze_encoder()

# 解冻 encoder（训练整个模型）
model.unfreeze_encoder()

# 检查可训练参数
params = model.get_num_params()
print(f"总参数: {params['total']:,}")
print(f"可训练参数: {params['trainable']:,}")
```

#### 4. 高级特征提取

```python
# 获取原始 encoder 输出
encoder_output = model.get_encoder_output(input_ids, attention_mask)
# Shape: [batch_size, seq_len, 768]

# 获取 pooled 表示
pooled = model.get_pooled_representation(input_ids, attention_mask)
# Shape: [batch_size, 768]

# 获取注意力权重（用于可视化）
attentions = model.compute_attention_weights(input_ids, attention_mask)
# 元组，每层一个注意力矩阵
```

## 训练策略

### 1. 全参数微调（推荐）

```python
# 保持默认设置，所有参数都可训练
model = CodeBertDetector()

# 训练时使用较小的学习率（2e-5 ~ 5e-5）
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

**特点：** 充分利用预训练知识，需要更多计算资源。

### 2. Encoder 冻结（快速实验）

```python
# 冻结 encoder，只训练分类头
model = CodeBertDetector(freeze_encoder=True)

# 使用更大的学习率（1e-4 ~ 1e-3）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

**特点：** 训练速度快，但性能可能下降。

### 3. 分层学习率（高级）

```python
# 不同层使用不同学习率
def get_layerwise_lr(model, base_lr=1e-5, multiplier=1.5):
    param_groups = []
    
    # encoder 参数按层分组
    for layer_idx in range(12):  # CodeBERT 有 12 层
        layer_lr = base_lr * (multiplier ** (11 - layer_idx))
        
        for name, param in model.encoder.named_parameters():
            if f"encoder.layer.{layer_idx}" in name:
                param_groups.append({
                    'params': [param],
                    'lr': layer_lr
                })
    
    # 分类头使用最高学习率
    for name, param in model.named_parameters():
        if 'head' in name:
            param_groups.append({
                'params': [param],
                'lr': base_lr * (multiplier ** 12)
            })
    
    return param_groups

optimizer = torch.optim.AdamW(get_layerwise_lr(model), weight_decay=1e-4)
```

**特点：** 更高层（更通用）学习率较低，低层（更特定）学习率较高。

## 模型参数

### CodeBERT-Base

| 参数 | 值 | 说明 |
|---|---|---|
| 层数 | 12 | Transformer encoder layers |
| 注意力头数 | 12 | Self-attention heads |
| 隐藏维度 | 768 | Hidden size |
| 中间维度 | 3072 | FFN intermediate size |
| 词表大小 | 50,265 | RoBERTa tokenizer |
| 总参数 | 125M | 不含分类头 |

### 额外分类头

| 组件 | 参数数 | 说明 |
|---|---|---|
| binary_head | 1,538 | Linear(768 → 2) |
| type_head | 4,614 | Linear(768 → 6) |
| 总计 | ~126.5M | 含 encoder |

## 模型输入输出

### 输入格式

```python
{
    'input_ids': torch.LongTensor,      # [batch_size, seq_len]
    'attention_mask': torch.LongTensor  # [batch_size, seq_len]
}
```

### 输出格式（训练）

```python
{
    'binary_logits': torch.FloatTensor,   # [batch_size, 2]
    'type_logits': torch.FloatTensor      # [batch_size, 6]
}
```

### 输出格式（推理）

```python
{
    'binary_probs': torch.FloatTensor,    # [batch_size, 2]，和为 1
    'type_probs': torch.FloatTensor,      # [batch_size, 6]，和为 1
    'binary_pred': torch.LongTensor,      # [batch_size]，值 ∈ {0, 1}
    'type_pred': torch.LongTensor         # [batch_size]，值 ∈ {0-5}
}
```

## 模型可替换性设计

系统架构支持灵活替换 encoder，只需改动一行代码：

### 替换为 GraphCodeBERT

```python
class CodeBertDetector(nn.Module):
    def __init__(self, model_name="microsoft/graphcodebert-base", ...):
        # GraphCodeBERT：更好地处理代码结构
        self.encoder = AutoModel.from_pretrained(model_name)
        ...
```

### 替换为 UniXcoder

```python
class CodeBertDetector(nn.Module):
    def __init__(self, model_name="microsoft/unixcoder-base", ...):
        # UniXcoder：支持多语言代码
        self.encoder = AutoModel.from_pretrained(model_name)
        ...
```

### 替换为自定义 Encoder

```python
class CodeBertDetector(nn.Module):
    def __init__(self, encoder_model, ...):
        # 支持任意兼容的 encoder
        self.encoder = encoder_model
        ...
```

## 使用示例

### 示例 1：基础使用

```python
import torch
from models.codebert_detector import CodeBertDetector
from transformers import RobertaTokenizer

# 初始化
model = CodeBertDetector()
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model.eval()

# 推理
code = "eval(input())"
inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    binary_logits = outputs['binary_logits']
    
    binary_probs = torch.softmax(binary_logits, dim=1)
    prediction = torch.argmax(binary_logits, dim=1)

print(f"安全概率: {binary_probs[0, 0]:.4f}")
print(f"恶意概率: {binary_probs[0, 1]:.4f}")
print(f"预测: {'恶意' if prediction[0] == 1 else '安全'}")
```

### 示例 2：保存与加载

```python
# 保存
torch.save(model.state_dict(), 'model.pt')
model.encoder.save_pretrained('encoder/')

# 加载
model = CodeBertDetector()
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

### 示例 3：特征可视化

```python
# 获取 hidden 表示用于可视化/分析
outputs = model(input_ids, attention_mask, return_hidden=True)
hidden = outputs['hidden']  # [batch_size, 768]

# 使用 t-SNE 或 PCA 可视化
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
hidden_2d = tsne.fit_transform(hidden.cpu().numpy())

# 绘制
import matplotlib.pyplot as plt
plt.scatter(hidden_2d[:, 0], hidden_2d[:, 1])
plt.show()
```

## 性能基准

在标准测试集上的性能（参考）：

| 指标 | 值 | 说明 |
|---|---|---|
| Accuracy | 0.92 | 整体准确率 |
| Precision | 0.88 | 恶意类精确率 |
| Recall | 0.95 | 恶意类召回率 |
| F1-Score | 0.91 | 调和平均 |
| ROC-AUC | 0.94 | ROC 曲线下面积 |

## 常见问题

### Q1: 如何减少模型大小？

答：考虑以下方法：
1. 知识蒸馏（使用更小的 student 模型）
2. 模型量化（转换为 INT8）
3. 使用 DistilBERT 或更小的预训练模型

### Q2: 如何处理长代码序列？

答：CodeBERT 支持最长 512 tokens。对于更长代码：
1. 截断关键部分
2. 分段处理 + 聚合
3. 使用支持更长序列的模型（如 LongFormer）

### Q3: 是否支持多 GPU？

答：支持，使用 `nn.DataParallel` 或 `DistributedDataParallel`：
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Q4: 如何微调已保存的模型？

答：
```python
# 加载已保存的模型
model = CodeBertDetector()
model.load_state_dict(torch.load('model.pt'))

# 继续训练
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# ...训练循环...
```

## 相关文档

- `../training/README.md` - 训练流程
- `../inference/README.md` - 推理与部署
- `../config.py` - 配置参考
