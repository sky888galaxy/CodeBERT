"""流量模态 v1：行为级恶意检测训练脚本。

使用通用流量特征空间训练 MLP，不为特定数据集改动模型结构，便于后续扩展更多数据集或更复杂模型（如时序 Transformer）。
"""

from __future__ import annotations

import argparse
import json
import pickle
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from traffic_detector.config import CHECKPOINT_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from traffic_detector.features.flow_schema import (
    SPEC_VERSION,
    get_default_flow_feature_spec,
)
from traffic_detector.features.flow_transform import FlowFeatureTransformer
from traffic_detector.models.flow_mlp import FlowMLPDetector


class FlowDataset(Dataset):
    """简单的张量数据集封装。"""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:  # noqa: D401
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def parse_hidden_dims(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def train_one_epoch(
    model: FlowMLPDetector,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.train()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)

        optimizer.zero_grad()
        logits, _ = model(batch_x, return_logits=True)
        logits = logits.squeeze(1)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, acc, f1, mcc


@torch.no_grad()
def evaluate(
    model: FlowMLPDetector,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.float().to(device)
        logits, _ = model(batch_x, return_logits=True)
        logits = logits.squeeze(1)
        loss = loss_fn(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_y.cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, acc, f1, mcc


def main() -> None:
    parser = argparse.ArgumentParser(description="流量模态 v1 行为级恶意检测 - MLP 训练")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=str, default="256,256,128,64")
    parser.add_argument("--patience", type=int, default=5, help="验证指标连续多少个 epoch 未提升则提前停止")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}, epochs={args.epochs}, batch_size={args.batch_size}", flush=True)

    feature_spec = get_default_flow_feature_spec()
    transformer = FlowFeatureTransformer(feature_spec=feature_spec)

    attack_path = RAW_DATA_DIR / "dataset_attack.csv"
    normal_path = RAW_DATA_DIR / "dataset_normal.csv"
    print(f"[data] loading CSVs: attack={attack_path}, normal={normal_path}", flush=True)
    X, y = transformer.fit_transform_from_csv(str(attack_path), str(normal_path))
    print(f"[data] finished feature build: X shape={X.shape}, y shape={y.shape}", flush=True)

    expected_dim = len(feature_spec.feature_names)
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"特征维度不一致: got {X.shape[1]}, expected {expected_dim}. 请检查 feature_spec 顺序与映射。"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"[split] train={X_train.shape[0]} rows, val={X_val.shape[0]} rows, input_dim={expected_dim}",
        flush=True,
    )

    train_dataset = FlowDataset(X_train, y_train)
    val_dataset = FlowDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    hidden_dims = parse_hidden_dims(args.hidden_dims)
    model = FlowMLPDetector(input_dim=expected_dim, hidden_dims=hidden_dims).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(
        f"[model] FlowMLPDetector initialized. hidden_dims={hidden_dims}, params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    # 早停追踪：以验证集 MCC 为准，防止 F1 对类不平衡过于敏感。
    best_metric = float("-inf")
    no_improve_epochs = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(transformer.scaler, f)

    MODEL_NAME = "FlowMLPDetector"
    MODEL_VERSION = "v1"

    history = []

    print("[train] start training loop", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, train_mcc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc, val_f1, val_mcc = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | "
            f"Val F1={val_f1:.4f} | Val MCC={val_mcc:.4f}"
        )

        # 记录完整指标，便于后续在 Jupyter 中可视化
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "train_mcc": train_mcc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_mcc": val_mcc,
            }
        )

        val_metric = val_mcc  # 早停指标：验证集 MCC

        if val_metric > best_metric:
            best_metric = val_metric
            no_improve_epochs = 0
            ckpt_path = CHECKPOINT_DIR / "flow_mlp_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": expected_dim,
                    "hidden_dims": hidden_dims,
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "spec_version": SPEC_VERSION,
                },
                ckpt_path,
            )

            cfg_path = CHECKPOINT_DIR / "flow_mlp_config.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_names": feature_spec.feature_names,
                        "spec_version": SPEC_VERSION,
                        "model_name": MODEL_NAME,
                        "model_version": MODEL_VERSION,
                        "hidden_dims": hidden_dims,
                        "checkpoint": str(ckpt_path),
                        "scaler_path": str(scaler_path),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"[Checkpoint] Saved best model to {ckpt_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(
                    f"[EarlyStopping] 验证集指标 {args.patience} 个 epoch 未提升，"
                    f"在第 {epoch} 轮提前停止。"
                )
                break

    # 将训练日志落盘，方便后续在 Jupyter 可视化（每次训练覆盖，表示最近一次完整记录）
    history_path = CHECKPOINT_DIR / "flow_mlp_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 训练脚本摘要：
    # - 早停指标：验证集 MCC
    # - 默认 patience：5
    # - 连续 patience 个 epoch 指标未提升即提前停止，仍保留最佳 checkpoint


if __name__ == "__main__":
    main()
