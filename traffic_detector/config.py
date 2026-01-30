"""流量检测项目的全局配置。

本模块定义数据与模型相关的基础路径，供训练、评估与推理脚本统一引用。
"""
from pathlib import Path

# 根目录以当前文件所在目录为基准，方便项目独立运行。
ROOT_DIR: Path = Path(__file__).resolve().parent

# 数据与中间产物目录配置。
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# 模型检查点保存目录，供训练和推理共用。
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"

__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "CHECKPOINT_DIR",
]
