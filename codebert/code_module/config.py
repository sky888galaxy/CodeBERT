"""
全局配置文件
包含模型路径、超参数、类型映射等所有配置项
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "code_module" / "data"
MODEL_DIR = BASE_DIR / "models" / "codebert-base"  # 本地离线权重
MODEL_SAVE_DIR = BASE_DIR / "experiments" / "runs"
RESULTS_DIR = BASE_DIR / "experiments" / "results"

# 数据文件路径
LABELED_DATASET = DATA_DIR / "labeled_dataset.json"
FINAL_DATASET = DATA_DIR / "final_dataset.json"
CVE_SAMPLES = DATA_DIR / "cve_samples.json"
SAFE_SAMPLES = DATA_DIR / "safe_samples.json"

# ==================== 模型配置 ====================
MODEL_NAME = str(MODEL_DIR)  # 只能从本地加载
MAX_LENGTH = 256
HIDDEN_SIZE = 768  # CodeBERT hidden size

# ==================== 类型映射 ====================
# 恶意代码类型定义（当前阶段）
TYPE_LABELS = {
    0: "Normal",              # 正常代码
    1: "WebShell",            # WebShell 后门
    2: "CommandInjection",    # 命令注入
    3: "Backdoor",            # 后门/木马
    4: "OtherMalicious"       # 其他恶意代码
}

NUM_TYPES = len(TYPE_LABELS)
TYPE_TO_ID = {v: k for k, v in TYPE_LABELS.items()}

# 二分类标签
BINARY_LABELS = {0: "Safe", 1: "Malicious"}

# ==================== 训练超参 ====================
class TrainingConfig:
    # 基础超参
    batch_size = 16
    learning_rate = 5e-5
    weight_decay = 1e-4
    max_epochs = 15
    gradient_accumulation_steps = 1  # 梯度累积（显存不足时提升）
    max_grad_norm = 1.0
    
    # Scheduler
    warmup_ratio = 0.1  # warmup steps = warmup_ratio * total_steps
    scheduler_type = "linear"  # linear, cosine, constant
    
    # Early Stopping
    early_stopping_patience = 3
    early_stopping_metric = "val_f1"  # val_loss, val_f1, val_roc_auc
    
    # 数据切分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random_seed = 42
    
    # 多任务权重
    binary_loss_weight = 1.0
    type_loss_weight = 1.0  # λ in L = L_binary + λ * L_type
    
    # 弱标注样本权重
    strong_sample_weight = 1.0
    weak_sample_weight = 0.4  # 弱标注样本的损失降权
    
    # Loss 类型
    use_focal_loss = True  # True: Focal Loss, False: CrossEntropy
    focal_alpha = None  # 自动根据类别分布计算，或手动设置 [alpha_0, alpha_1]
    focal_gamma = 2.5
    
    # 类别不平衡处理
    use_class_weights = True  # 是否使用类别权重
    
    # 日志与保存
    log_every_n_steps = 50
    save_best_only = True
    save_checkpoint_every_n_epochs = 1

# ==================== 推理配置 ====================
class InferenceConfig:
    # 拒识区间（置信度阈值）
    uncertainty_threshold_low = 0.3
    uncertainty_threshold_high = 0.7
    
    # 置信度等级划分
    # certainty = "high" if max_prob > 0.85 else "medium" if > 0.65 else "low"
    high_confidence_threshold = 0.85
    medium_confidence_threshold = 0.65
    
    # 决策阈值（用于二分类）
    binary_decision_threshold = 0.5

# ==================== 设备配置 ====================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # DataLoader workers

# ==================== 辅助函数 ====================
def get_type_name(type_id):
    """根据 type_id 获取类型名称"""
    return TYPE_LABELS.get(type_id, "Unknown")

def get_type_id(type_name):
    """根据类型名称获取 type_id"""
    return TYPE_TO_ID.get(type_name, -1)

def ensure_dirs():
    """确保所有必要目录存在"""
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("=" * 50)
    print("配置信息")
    print("=" * 50)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"DEVICE: {DEVICE}")
    print(f"TYPE_LABELS: {TYPE_LABELS}")
    print(f"TrainingConfig.batch_size: {TrainingConfig.batch_size}")
    print(f"TrainingConfig.learning_rate: {TrainingConfig.learning_rate}")
