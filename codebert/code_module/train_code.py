"""
Offline multitask training script for CodeBERT detector.
- Loads data from code_module/data/train.json, val.json, test.json (list or jsonl).
- Uses local CodeBERT encoder with dual heads (binary + type).
- Supports weighted CE or Focal loss, weak-label down-weighting, class weights, gradient accumulation,
  warmup scheduler, and early stopping on validation F1.
Run from repo root:
    python code_module/train_code.py
"""

import os
import sys

# 计算项目根目录（code_module 的上一层），并确保在 sys.path 中用于脚本直接运行时的绝对导入
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from common.config import (
    DATA_DIR,
    MODEL_SAVE_DIR,
    RESULTS_DIR,
    TYPE_TO_ID,
    TYPE_LABELS,
    MAX_LENGTH,
    TrainingConfig,
    DEVICE,
    LOCAL_MODEL_PATH,
)
from common.metrics import compute_metrics
from common.utils import set_seed
from code_module.models.codebert_detector import CodeBertDetector
from code_module.training.losses import FocalLoss, WeightedCrossEntropyLoss

CHECKPOINT_DIR = Path("code_module") / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json_or_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset, got {type(data)}")
    return data


def map_type_label(val) -> Tuple[int, bool]:
    # returns (type_id, valid_mask)
    if isinstance(val, int):
        return val, val >= 0
    if isinstance(val, str):
        return TYPE_TO_ID.get(val, -1), TYPE_TO_ID.get(val, -1) >= 0
    return -1, False


class CodeDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int, weak_weight: float, strong_weight: float):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.weak_weight = weak_weight
        self.strong_weight = strong_weight

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        code = sample.get("code", "")
        binary_label = int(sample.get("binary_label", sample.get("label", 0)))
        raw_type = sample.get("type_label", None)
        type_id, valid_type = map_type_label(raw_type)
        label_type = sample.get("label_type", "strong")

        encoded = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        sample_weight = self.weak_weight if str(label_type).lower() == "weak" else self.strong_weight

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "type_label": torch.tensor(type_id, dtype=torch.long),
            "type_mask": torch.tensor(1 if valid_type else 0, dtype=torch.float),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float),
        }


def collate_fn(batch):
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        collated[k] = torch.stack([item[k] for item in batch], dim=0)
    return collated


def build_loaders(tokenizer, args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_data = load_json_or_jsonl(Path(args.train_path))
    val_data = load_json_or_jsonl(Path(args.val_path))
    test_data = load_json_or_jsonl(Path(args.test_path))

    train_ds = CodeDataset(train_data, tokenizer, args.max_length, args.weak_weight, args.strong_weight)
    val_ds = CodeDataset(val_data, tokenizer, args.max_length, args.weak_weight, args.strong_weight)
    test_ds = CodeDataset(test_data, tokenizer, args.max_length, args.weak_weight, args.strong_weight)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def compute_class_weights(train_loader, device):
    binary_counts = torch.zeros(2, dtype=torch.float)
    type_counts = torch.zeros(len(TYPE_LABELS), dtype=torch.float)
    for batch in train_loader:
        lbl = batch["binary_label"]
        for v in lbl:
            binary_counts[int(v)] += 1
        t_lbl = batch["type_label"]
        t_mask = batch["type_mask"]
        for val, m in zip(t_lbl, t_mask):
            if m.item() > 0 and int(val) >= 0 and int(val) < len(TYPE_LABELS):
                type_counts[int(val)] += 1
    # avoid div zero
    binary_weights = 1.0 / torch.clamp(binary_counts, min=1.0)
    binary_weights = binary_weights / binary_weights.sum() * 2
    type_counts = torch.where(type_counts > 0, type_counts, torch.ones_like(type_counts))
    type_weights = 1.0 / type_counts
    type_weights = type_weights / type_weights.sum() * len(TYPE_LABELS)
    return binary_weights.to(device), type_weights.to(device)


def build_criterion(args, binary_weights, type_weights):
    if args.use_focal:
        binary_loss_fn = FocalLoss(alpha=binary_weights, gamma=args.focal_gamma, reduction="none")
        type_loss_fn = FocalLoss(alpha=type_weights, gamma=args.focal_gamma, reduction="none")
    else:
        binary_loss_fn = WeightedCrossEntropyLoss(class_weights=binary_weights, reduction="none")
        type_loss_fn = WeightedCrossEntropyLoss(class_weights=type_weights, reduction="none")
    return binary_loss_fn, type_loss_fn


def get_scheduler(optimizer, num_training_steps, args):
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)


def evaluate(model, loader, device) -> Dict:
    model.eval()
    all_bin_labels, all_bin_preds, all_bin_probs = [], [], []
    all_type_labels, all_type_preds, all_type_probs = [], [], []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_label = batch["binary_label"].to(device)
            type_label = batch["type_label"].to(device)
            type_mask = batch["type_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            binary_logits = outputs["binary_logits"]
            type_logits = outputs["type_logits"]

            binary_probs = torch.softmax(binary_logits, dim=1)
            binary_pred = torch.argmax(binary_probs, dim=1)
            all_bin_labels.append(binary_label.cpu())
            all_bin_preds.append(binary_pred.cpu())
            all_bin_probs.append(binary_probs.cpu())

            # type metrics only on valid labels
            valid_mask = type_mask.bool()
            if valid_mask.any():
                t_logits = type_logits[valid_mask]
                t_label = type_label[valid_mask]
                t_probs = torch.softmax(t_logits, dim=1)
                t_pred = torch.argmax(t_probs, dim=1)
                all_type_labels.append(t_label.cpu())
                all_type_preds.append(t_pred.cpu())
                all_type_probs.append(t_probs.cpu())

    bin_labels = torch.cat(all_bin_labels) if all_bin_labels else torch.tensor([], dtype=torch.long)
    bin_preds = torch.cat(all_bin_preds) if all_bin_preds else torch.tensor([], dtype=torch.long)
    bin_probs = torch.cat(all_bin_probs) if all_bin_probs else torch.tensor([], dtype=torch.float)

    metrics = {
        "binary": compute_metrics(bin_labels, bin_preds, bin_probs, num_classes=2, task_name="binary"),
    }

    if all_type_labels:
        t_labels = torch.cat(all_type_labels)
        t_preds = torch.cat(all_type_preds)
        t_probs = torch.cat(all_type_probs)
        metrics["type"] = compute_metrics(t_labels, t_preds, t_probs, num_classes=len(TYPE_LABELS), task_name="multiclass")
    else:
        metrics["type"] = {"note": "no valid type labels"}

    return metrics


def train(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    train_loader, val_loader, test_loader = build_loaders(tokenizer, args)

    model = CodeBertDetector(model_path=args.model_path, num_types=len(TYPE_LABELS), dropout_prob=args.dropout, max_length=args.max_length, load_tokenizer=False)
    model.to(DEVICE)

    binary_weights, type_weights = compute_class_weights(train_loader, DEVICE) if args.use_class_weights else (
        torch.ones(2, device=DEVICE), torch.ones(len(TYPE_LABELS), device=DEVICE)
    )
    binary_loss_fn, type_loss_fn = build_criterion(args, binary_weights, type_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs
    scheduler = get_scheduler(optimizer, total_steps, args)

    best_f1 = -1.0
    best_path = CHECKPOINT_DIR / "best.pt"
    log_records = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            binary_label = batch["binary_label"].to(DEVICE)
            type_label = batch["type_label"].to(DEVICE)
            type_mask = batch["type_mask"].to(DEVICE)
            sample_weight = batch["sample_weight"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            binary_logits = outputs["binary_logits"]
            type_logits = outputs["type_logits"]

            # binary loss: per-sample then weighted average
            b_loss_vec = binary_loss_fn(binary_logits, binary_label)  # [B]
            b_weight = sample_weight  # [B]
            b_loss = (b_loss_vec * b_weight).sum() / b_weight.sum().clamp_min(1e-8)

            # type loss: filter valid labels first, then compute on filtered samples only
            # (avoids CUDA assertion from invalid type labels being passed to CrossEntropyLoss)
            valid_mask = type_mask.bool()
            if valid_mask.any():
                # Filter to valid samples BEFORE passing to loss function
                t_logits_valid = type_logits[valid_mask]            # [V, num_types]
                t_labels_valid = type_label[valid_mask]             # [V]
                t_weight = sample_weight[valid_mask]                # [V]
                t_loss_vec = type_loss_fn(t_logits_valid, t_labels_valid)  # [V] - only valid samples
                t_loss = (t_loss_vec * t_weight).sum() / t_weight.sum().clamp_min(1e-8)
            else:
                t_loss = torch.tensor(0.0, device=DEVICE)

            loss = b_loss + args.lambda_type * t_loss
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # validation
        val_metrics = evaluate(model, val_loader, DEVICE)
        val_f1 = val_metrics["binary"].get("f1", 0.0) or 0.0

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_binary_f1": val_f1,
            "val_metrics": val_metrics,
        }
        log_records.append(log_entry)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"[OK] Saved best model to {best_path} (val_f1={val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1 if 'patience_counter' in locals() else 1
            if patience_counter >= args.early_stop:
                print("Early stopping triggered.")
                break

    # final test eval on best model
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_metrics = evaluate(model, test_loader, DEVICE)
    print("Test metrics:", json.dumps(test_metrics, indent=2, ensure_ascii=False))

    # write log
    log_file = RESULTS_DIR / "train_log.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump({"logs": log_records, "best_f1": best_f1, "test_metrics": test_metrics}, f, ensure_ascii=False, indent=2)
    print(f"Logs saved to {log_file}")
    if best_path.exists():
        print(f"Best checkpoint: {best_path}")


def parse_args():
    cfg = TrainingConfig()
    parser = argparse.ArgumentParser(description="Offline multitask training for CodeBERT detector")
    parser.add_argument("--model-path", type=Path, default=Path(str(LOCAL_MODEL_PATH)))
    parser.add_argument("--train-path", type=Path, default=Path(cfg.train_path))
    parser.add_argument("--val-path", type=Path, default=Path(cfg.val_path))
    parser.add_argument("--test-path", type=Path, default=Path(cfg.test_path))

    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.max_epochs)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-accum-steps", type=int, default=cfg.gradient_accumulation_steps)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.warmup_ratio)
    parser.add_argument("--scheduler", type=str, default=cfg.scheduler_type, choices=["linear", "cosine"])
    parser.add_argument("--max-grad-norm", type=float, default=cfg.max_grad_norm)

    parser.add_argument("--use-focal", action="store_true", default=cfg.use_focal_loss)
    parser.add_argument("--focal-gamma", type=float, default=cfg.focal_gamma)
    parser.add_argument("--use-class-weights", action="store_true", default=cfg.use_class_weights)
    parser.add_argument("--lambda-type", type=float, default=cfg.lambda_type)
    parser.add_argument("--weak-weight", type=float, default=cfg.weak_weight)
    parser.add_argument("--strong-weight", type=float, default=cfg.strong_weight)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--early-stop", type=int, default=cfg.early_stopping_patience)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
