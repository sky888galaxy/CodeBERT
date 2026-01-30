"""
Common configuration for offline CodeBERT training/inference.
Everything must work without network access. Model/tokenizer are loaded from local path ./models/codebert-base.
"""
from pathlib import Path
import torch

# Project root (codebert/)
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_MODULE_DIR = BASE_DIR / "code_module"
DATA_DIR = CODE_MODULE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "codebert-base"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODEL_SAVE_DIR = EXPERIMENTS_DIR / "runs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Type labels mapping
TYPE_LABELS = {
    0: "Normal",
    1: "WebShell",
    2: "CommandInjection",
    3: "Backdoor",
    4: "OtherMalicious",
}
NUM_TYPES = len(TYPE_LABELS)
TYPE_TO_ID = {v: k for k, v in TYPE_LABELS.items()}
BINARY_LABELS = {0: "Safe", 1: "Malicious"}

# Default tokenizer/model name (local path only)
LOCAL_MODEL_PATH = MODEL_DIR
MAX_LENGTH = 256  # shorter for memory; can override in TrainingConfig
HIDDEN_SIZE = 768

class TrainingConfig:
    # data
    train_path = DATA_DIR / "train.json"
    val_path = DATA_DIR / "val.json"
    test_path = DATA_DIR / "test.json"

    # hyperparameters
    batch_size = 16
    max_epochs = 12
    learning_rate = 5e-5
    weight_decay = 1e-4
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1

    # scheduler
    scheduler_type = "linear"  # linear | cosine
    warmup_ratio = 0.1  # 10% warmup

    # multitask
    lambda_type = 1.0  # total_loss = Lb + lambda_type * Lt
    use_focal_loss = True
    focal_gamma = 2.0

    # imbalance handling
    use_class_weights = True

    # weak label weighting
    weak_weight = 0.4
    strong_weight = 1.0

    # early stopping
    early_stopping_patience = 3
    monitor_metric = "val_f1"
    monitor_mode = "max"

    # logging
    log_every_n_steps = 50

    # random seed
    seed = 42

class InferenceConfig:
    uncertainty_low = 0.3
    uncertainty_high = 0.7
    high_confidence = 0.85
    medium_confidence = 0.65

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_type_name(tid: int) -> str:
    return TYPE_LABELS.get(int(tid), "Unknown")


def get_type_id(name: str) -> int:
    return TYPE_TO_ID.get(name, -1)
