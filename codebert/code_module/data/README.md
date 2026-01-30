# 数据集目录说明

## 概述
本目录包含 CodeBERT 恶意代码检测系统的所有数据集及相关脚本。

## 文件结构

```
data/
├── README.md                    # 本文件
├── split_dataset.py             # 数据切分脚本
├── labeled_dataset.json         # 原始标注数据集（来自 code_module 上级目录）
├── train.json                   # 训练集（自动生成）
├── val.json                     # 验证集（自动生成）
├── test.json                    # 测试集（自动生成）
└── split_summary.json           # 切分摘要与统计信息
```

## 数据格式规范

### 原始数据格式（向后兼容）
```json
{
  "code": "...代码内容...",
  "label": 0,                      // 旧格式：0=安全，1=恶意
  "source": "2016/CVE-2016-10034",
  "malicious_type": ["WebShell"]   // 可选
}
```

### 规范化格式（新格式）
```json
{
  "code": "...代码内容...",
  "binary_label": 0,               // 必须：0=安全，1=恶意
  "type_label": 1,                 // 可选：0=Normal, 1=WebShell, 2=CommandInjection, ...
  "label_type": "strong",          // 必须："strong" 或 "weak"
  "source": "2016/CVE-2016-10034", // 来源追溯
  "confidence": 0.95               // 可选：标注者置信度
}
```

### 类型标签映射
| ID | 名称 | 说明 |
|---|---|---|
| 0 | Normal | 正常代码 |
| 1 | WebShell | WebShell/后门 |
| 2 | CommandInjection | 命令注入 |
| 3 | SQLInjection | SQL 注入 |
| 4 | XSS | 跨站脚本 |
| 5 | OtherMalicious | 其他恶意代码 |

## 使用流程

### 1. 数据切分

如果尚未进行数据切分，执行以下命令：

```bash
python split_dataset.py \
  --input ../labeled_dataset.json \
  --output . \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

**参数说明：**
- `--input`: 原始数据文件路径
- `--output`: 输出目录（生成 train.json, val.json, test.json）
- `--train-ratio`: 训练集比例（默认 0.8）
- `--val-ratio`: 验证集比例（默认 0.1）
- `--test-ratio`: 测试集比例（默认 0.1）
- `--seed`: 随机种子，确保可复现（默认 42）

### 2. 数据切分特性

✅ **分层抽样**：按二分类标签进行分层，确保各集合的类别分布相近

✅ **防穿越污染**：使用固定随机种子 + 自动验证，确保 train/val/test 间无样本重叠

✅ **可复现性**：同一随机种子保证每次切分结果完全相同

✅ **自动统计**：生成 split_summary.json 记录详细统计信息

### 3. 弱标注样本处理

标注数据时，若样本置信度不足或来自自动标注（如主动学习），应设置：
```json
{
  "code": "...",
  "binary_label": 1,
  "type_label": -1,              // -1 表示无有效类型标签
  "label_type": "weak"            // 标记为弱标注
}
```

**在训练中的处理：**
- 弱标注样本的损失权重自动降低为 0.4（相对强标注 1.0）
- 无有效 type_label 的样本只参与二分类任务，不计算类型分类损失

### 4. 主动学习数据融合

新增的主动学习样本应遵循以下原则：

1. **仅追加到 train.json**，不修改 val.json 或 test.json
2. **标记为弱标注**：`"label_type": "weak"`
3. **不覆盖现有数据**
4. **定期重新执行完整训练**以利用新数据

**示例脚本：**
```python
import json

# 加载现有训练集
with open('train.json', 'r') as f:
    train_data = json.load(f)

# 加载新的主动学习样本
with open('active_learning_samples.json', 'r') as f:
    new_samples = json.load(f)

# 标记新样本为弱标注
for sample in new_samples:
    sample['label_type'] = 'weak'
    if 'type_label' not in sample:
        sample['type_label'] = -1  # 无有效类型标签

# 合并
train_data.extend(new_samples)

# 保存
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)
```

## 数据统计示例

执行 split_dataset.py 后，会生成如下形式的统计信息：

```
数据集切分摘要:
  总样本数: 15000
  训练集: 12000 (80.0%)
  验证集: 1500 (10.0%)
  测试集: 1500 (10.0%)

各集合的二分类分布:
  train: 正常  4500 (37.5%) | 恶意  7500 (62.5%)
  val:   正常   550 (36.7%) | 恶意   950 (63.3%)
  test:  正常   550 (36.7%) | 恶意   950 (63.3%)
```

## 常见问题

### Q1: 如何处理类别不平衡？

答：系统在训练中会自动计算类别权重，恶意代码样本较少时会自动提高其权重。详见 `training/losses.py` 中的 `compute_class_weights()` 函数。

### Q2: 如何区分强标注和弱标注？

答：在样本 JSON 中设置 `label_type` 字段：
- `"label_type": "strong"` - 由专家手工标注，置信度高
- `"label_type": "weak"` - 由自动系统标注或置信度不足

弱标注样本的训练损失会被自动降权至 0.4 倍。

### Q3: 如果数据文件很大会怎样？

答：建议使用分页加载或流式处理。可修改 `dataset.py` 中的 `load_data_from_json()` 添加流式加载逻辑。

### Q4: 是否支持在线学习或持续更新？

答：支持。将新数据追加到 `train.json`，重新运行训练脚本即可。系统会自动利用新数据进行微调。

## 版本历史

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2025-12-07 | 初始版本，支持多任务学习 + 弱标注区分 |

## 贡献指南

在修改或扩展数据处理逻辑时，请遵循以下原则：

1. 保持数据格式一致
2. 添加新字段时需要向后兼容
3. 更新本 README 文档
4. 运行数据验证脚本确保数据完整性

## 相关文档

- `../training/README.md` - 训练流程说明
- `../inference/README.md` - 推理与部署说明
- `../models/README.md` - 模型架构详解
