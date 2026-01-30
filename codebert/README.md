CodeBERT 多任务恶意代码检测（离线版）
======================================

一个离线可复现的代码安全检测框架，包含数据准备、训练、评估与推理全流程，默认仅使用本地模型 `./models/codebert-base`，适配 `TRANSFORMERS_OFFLINE=1`。

项目结构（核心）
---------------
- `code_module/config.py` 全局配置（路径、超参、类型映射）
- `code_module/data/prepare_data.py` 合并本地数据为 `combined_dataset.json`
- `code_module/data/split_dataset.py` 分层/分库切分生成 `train/val/test.json`
- `code_module/models/codebert_detector.py` 本地 CodeBERT + 二分类/类型双头
- `code_module/train_code.py` 离线多任务训练脚本（CE/Focal，类权重，弱标注降权，早停，scheduler，grad accum）
- `code_module/inference/` 推理脚本与接口
- `experiments/runs` / `experiments/results` 模型与日志输出目录

快速开始（本地离线）
-------------------
1) 合并数据（可选，如已生成可跳过）
```
python code_module/data/prepare_data.py
```

2) 切分数据（确保生成 train/val/test）
```
python code_module/data/split_dataset.py --input code_module/data/combined_dataset.json --seed 42
```

3) 训练（多任务、离线）
```
python code_module/train_code.py \
  --batch-size 16 --epochs 12 --lr 5e-5 \
  --use-focal --focal-gamma 2.5 \
  --lambda-type 1.0 --weak-weight 0.4 \
  --grad-accum-steps 1 --scheduler linear --warmup-ratio 0.1
```
输出：`code_module/checkpoints/best.pt` 与 `experiments/results/train_log.json`。

4) 推理（加载同结构模型）
```
python code_module/inference/infer_code.py --checkpoint code_module/checkpoints/best.pt --mode batch
```

关键特性
--------
- 离线加载：本地 `./models/codebert-base`，无网络依赖。
- 多任务头：`binary_head` (2 类) + `type_head` (NUM_TYPES)。
- 损失与权重：类权重自动计算；CE/Focal 可选；弱标注降权；`lambda_type` 可调。
- 训练策略：AdamW，linear/cosine warmup scheduler，梯度累积，early stopping（监控 val F1），最佳模型 checkpoint。
- 评估：二分类 + 类型多分类指标，日志 JSON 便于绘图与对比。

常见路径与产物
--------------
- 数据：`code_module/data/{combined_dataset,train,val,test}.json`
- 模型：`code_module/checkpoints/best.pt`
- 日志：`experiments/results/train_log.json`

提示
----
- 显存不足：减小 batch_size，增大 `--grad-accum-steps`，或在 `codebert_detector` 启用 `freeze_encoder`。
- 类分布偏斜：保持 `--use-class-weights`，或调整 `--focal-gamma`/`--lambda-type`。
- 不确定样本挖掘：推理时关注概率落在 0.3~0.7 区间的样本，适合主动学习。
