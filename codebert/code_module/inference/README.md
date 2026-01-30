# 推理模块说明

## 概述

本模块提供高级推理接口，支持：
- 单样本推理
- 批量推理
- 文件推理
- 交互式推理
- 不确定性估计
- 置信度等级
- 拒识机制

## 文件结构

```
inference/
├── README.md                    # 本文件
└── infer_code.py                # 推理脚本与接口
```

## 快速开始

### 1. 交互式推理（推荐）

```bash
python infer_code.py \
  --model-path ../experiments/runs/best_model_20251207_120530 \
  --mode interactive \
  --verbose
```

效果示例：
```
======================================================================
CodeBERT 恶意代码检测结果
======================================================================

判断结果: ❌ 恶意代码
恶意概率: 0.9234
置信度: HIGH

恶意类型: WebShell

二分类概率分布:
  normal    : ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0766
  malicious : ████████████████████████████████████████████ 0.9234

各类型概率分布:
  WebShell         : ████████████████████████░░░░░░░░░░░░░░ 0.6234
  OtherMalicious   : ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.2156
  CommandInjection : ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.1089
  SQLInjection     : ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0401
  XSS              : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0120
  Normal           : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0000

======================================================================
```

### 2. 文件推理

```bash
python infer_code.py \
  --model-path ../experiments/runs/best_model_20251207_120530 \
  --mode file \
  --input /path/to/code.py \
  --verbose
```

### 3. 编程方式使用

```python
from inference.infer_code import CodeBertInferencer

# 初始化推理器
inferencer = CodeBertInferencer(
    model_path="../experiments/runs/best_model_20251207_120530"
)

# 单样本推理
result = inferencer.infer_single("""
def foo():
    import os
    os.system('rm -rf /')
""")

# 查看结果
print(f"恶意: {result['is_malicious']}")
print(f"概率: {result['binary_score']:.4f}")
print(f"类型: {result['type_label']}")
print(f"置信度: {result['certainty']}")

# 格式化输出
output = inferencer.format_result(result, verbose=True)
print(output)
```

## 核心数据结构

### 推理结果格式

```python
{
    # 基本信息
    'code': str,                    # 输入代码（截断至 200 字符）
    'is_malicious': bool,           # 是否为恶意代码
    
    # 二分类结果
    'binary_score': float,          # P(malicious) ∈ [0, 1]
    'binary_scores': {              # 详细概率
        'normal': float,
        'malicious': float
    },
    
    # 类型分类结果
    'type_label': str,              # 恶意类型名称
    'type_scores': {                # 各类型概率
        'Normal': float,
        'WebShell': float,
        'CommandInjection': float,
        'SQLInjection': float,
        'XSS': float,
        'OtherMalicious': float
    },
    
    # 置信度与不确定性
    'certainty': str,               # "high" / "medium" / "low"
    'confidence': float,            # 二分类最高概率
    'margin': float,                # 最高概率 - 次高概率
    'decision_uncertainty': str,    # "confident" / "uncertain" / "reject"
    
    # 类型信息
    'type_confidence': float,       # 类型分类最高概率
    'type_margin': float            # 类型分类 margin
}
```

## 关键概念

### 1. 置信度等级 (Certainty)

根据最高概率自动划分：

| 等级 | 条件 | 含义 |
|---|---|---|
| `high` | prob > 0.85 | 预测置信度非常高，可信度强 |
| `medium` | 0.65 < prob ≤ 0.85 | 预测有一定置信度，可接受 |
| `low` | prob ≤ 0.65 | 预测置信度较低，需谨慎 |

**阈值配置**（在 `config.py` 中）：
```python
class InferenceConfig:
    high_confidence_threshold = 0.85
    medium_confidence_threshold = 0.65
```

### 2. 决策不确定性 (Decision Uncertainty)

根据恶意概率划分：

| 状态 | 条件 | 含义 |
|---|---|---|
| `confident` | P(malicious) < 0.3 或 > 0.7 | 预测明确，不需人工审核 |
| `uncertain` | 0.3 ≤ P(malicious) ≤ 0.7 | **预测不确定，建议人工审核** |
| `reject` | 在拒识区间内 | 预测拒识，返回"无法判断" |

**拒识区间**（在 `config.py` 中）：
```python
class InferenceConfig:
    uncertainty_threshold_low = 0.3   # 下界
    uncertainty_threshold_high = 0.7  # 上界
```

### 3. Margin（决策裕度）

$$\text{Margin} = P(\text{最高类}) - P(\text{次高类})$$

**含义：**
- Margin 越大，二者差距越明显，预测更可信
- Margin 越小，二者接近，预测更不确定

**示例：**
```
情况 1: [0.9, 0.1]    -> margin = 0.8（非常可信）
情况 2: [0.55, 0.45]  -> margin = 0.1（不可信）
```

## 推理工作流

```
输入代码
  ↓
Tokenization（RobertaTokenizer）
  ↓
编码（[CLS] token pooling）
  ↓
CodeBERT Encoder
  ↓
二分类头 + 类型分类头
  ↓
Softmax → 概率分布
  ↓
置信度等级判断
  ↓
决策不确定性判断
  ↓
输出推理结果
```

## 使用示例

### 示例 1：基础推理

```python
from inference.infer_code import CodeBertInferencer

inferencer = CodeBertInferencer("../experiments/runs/best_model_20251207_120530")

code = """
import os
os.system('ls -la')
"""

result = inferencer.infer_single(code)

if result['decision_uncertainty'] == 'uncertain':
    print("⚠️  预测不确定，建议人工审核")
elif result['is_malicious']:
    print(f"❌ 检测到 {result['type_label']}")
else:
    print("✅ 代码看起来是安全的")
```

### 示例 2：批量推理

```python
codes = [
    "print('hello')",
    "os.system('rm -rf /')",
    "eval(input())",
    ...
]

results = inferencer.infer_batch(codes, batch_size=32)

# 统计
malicious_count = sum(1 for r in results if r['is_malicious'])
print(f"恶意代码: {malicious_count} / {len(codes)}")

# 高不确定样本用于主动学习
uncertain_samples = [r for r in results if r['decision_uncertainty'] == 'uncertain']
print(f"不确定样本（用于主动学习）: {len(uncertain_samples)}")
```

### 示例 3：与主动学习集成

```python
from inference.infer_code import CodeBertInferencer
import json

inferencer = CodeBertInferencer("../experiments/runs/best_model_20251207_120530")

# 候选代码池
candidate_codes = load_candidate_pool()

# 推理并选择不确定样本
uncertain_samples = []
for code in candidate_codes:
    result = inferencer.infer_single(code)
    
    if result['decision_uncertainty'] == 'uncertain':
        uncertain_samples.append({
            'code': code,
            'confidence': result['confidence'],
            'binary_score': result['binary_score'],
            'suggested_label': 'malicious' if result['binary_score'] > 0.5 else 'normal'
        })

# 排序（按不确定性降序）
uncertain_samples.sort(key=lambda x: abs(x['binary_score'] - 0.5))

# 保存不确定样本供标注
with open('uncertain_samples_for_annotation.json', 'w') as f:
    json.dump(uncertain_samples[:100], f, indent=2)  # 取 top 100

print(f"已选出 {len(uncertain_samples)} 个不确定样本用于标注")
```

### 示例 4：生成推理报告

```python
from inference.infer_code import CodeBertInferencer
from datetime import datetime

inferencer = CodeBertInferencer("../experiments/runs/best_model_20251207_120530")

# 扫描代码库
files = [...]  # 代码文件列表
results = []

for file_path in files:
    result = inferencer.infer_from_file(file_path)
    results.append(result)

# 生成报告
report = {
    'timestamp': datetime.now().isoformat(),
    'total_files': len(results),
    'malicious_files': sum(1 for r in results if r['is_malicious']),
    'high_confidence_detections': sum(1 for r in results if r['certainty'] == 'high' and r['is_malicious']),
    'uncertain_files': sum(1 for r in results if r['decision_uncertainty'] == 'uncertain'),
    'details': results
}

# 保存报告
import json
with open('scan_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"✅ 报告已生成: scan_report.json")
```

## 性能优化

### 1. 批量推理加速

对于大量代码，使用批量推理可以显著提高速度：

```python
# 单个推理: ~50ms/个
for code in codes:
    inferencer.infer_single(code)  # 缓慢

# 批量推理: ~0.5ms/个
results = inferencer.infer_batch(codes, batch_size=64)  # 快速
```

### 2. GPU 加速

确保 GPU 可用且模型在 GPU 上：

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inferencer = CodeBertInferencer(model_path, device=device)
```

### 3. 缓存与重用

长期运行的服务可以缓存推理器实例：

```python
# 全局推理器（避免重复加载）
_inferencer = None

def get_inferencer():
    global _inferencer
    if _inferencer is None:
        _inferencer = CodeBertInferencer("../experiments/runs/best_model")
    return _inferencer

# 在 Flask/FastAPI 服务中使用
@app.route('/detect', methods=['POST'])
def detect():
    code = request.json['code']
    inferencer = get_inferencer()
    result = inferencer.infer_single(code)
    return result
```

## 常见问题

### Q1: 如何降低误报率？

答：调整拒识区间阈值，增加 `uncertainty_threshold_high`：
```python
config.uncertainty_threshold_high = 0.8  # 提高至 0.8
```

### Q2: 推理速度如何？

答：在 GPU 上，单样本约 50ms，批量推理可达到 0.5ms/个。

### Q3: 是否支持实时/流式推理？

答：当前版本支持批量推理。可扩展为流式版本（推荐异步处理）。

### Q4: 如何在 Web 服务中部署？

答：参考 `示例 3` 的 Flask/FastAPI 集成示例。

## 相关文档

- `../training/README.md` - 训练流程
- `../data/README.md` - 数据管理
- `../models/README.md` - 模型架构
- `../config.py` - 配置参考
