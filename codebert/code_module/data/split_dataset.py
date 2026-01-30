"""
离线数据集切分脚本。

特点：
- 固定随机种子，结果可复现。
- 优先按 repo/project 级别切分，避免同一代码库跨集合（穿越污染）。
- 在分层上优先保证 binary_label 分布，同时尽量保持 type_label 覆盖。
- 默认比例：train 70%，val 15%，test 15%。

使用：
    python code_module/data/split_dataset.py
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ensure repo root (contains common/) is on sys.path for flexible CWD execution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.config import DATA_DIR, TYPE_LABELS, TYPE_TO_ID


DEFAULT_TRAIN = 0.7
DEFAULT_VAL = 0.15
DEFAULT_TEST = 0.15
DEFAULT_SEED = 42


def load_dataset(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    text = path.read_text(encoding="utf-8").strip()
    data: List[Dict] = []

    if text.startswith("["):
        data = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    if not isinstance(data, list):
        raise ValueError(f"期望 list, 但读取到 {type(data)}")
    print(f"✅ 加载 {len(data)} 条样本自 {path}")
    return data


def normalize_sample(sample: Dict) -> Dict:
    # 保留原字段，补齐缺省
    normalized = dict(sample)
    binary = normalized.get("binary_label", normalized.get("label", 0))
    normalized["binary_label"] = int(binary)

    type_val = normalized.get("type_label")
    if type_val is None:
        type_val = "Normal" if normalized["binary_label"] == 0 else "OtherMalicious"
    normalized["type_label"] = type_val

    label_type = normalized.get("label_type", "strong")
    normalized["label_type"] = label_type if label_type in ("strong", "weak") else "strong"
    return normalized


def choose_group_key(sample: Dict) -> str:
    # 优先使用 repo/project；否则尝试从 file_path 提取首段作为分组
    for key in ("repo_id", "project", "repo", "repository"):
        if key in sample and sample[key]:
            return str(sample[key])
    if "file_path" in sample and sample["file_path"]:
        path_str = str(sample["file_path"]).replace("\\", "/")
        return path_str.split("/")[0]
    return ""


def split_with_groups(samples: List[Dict], train_r: float, val_r: float, test_r: float, seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    total = len(samples)
    target_total = {
        "train": round(total * train_r),
        "val": round(total * val_r),
        "test": total - round(total * train_r) - round(total * val_r),
    }

    # 统计全局二分类期望
    global_binary = Counter(s["binary_label"] for s in samples)
    target_binary = {
        split: {lbl: round(global_binary[lbl] * ratio) for lbl in (0, 1)}
        for split, ratio in [("train", train_r), ("val", val_r), ("test", test_r)]
    }

    groups = defaultdict(list)
    ungrouped = []
    for s in samples:
        key = choose_group_key(s)
        if key:
            groups[key].append(s)
        else:
            ungrouped.append(s)

    split_data = {"train": [], "val": [], "test": []}
    split_bin_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    split_total = {"train": 0, "val": 0, "test": 0}

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)

    for _, g_samples in group_items:
        g_bin = Counter(s["binary_label"] for s in g_samples)
        majority_label = 1 if g_bin[1] >= g_bin[0] else 0

        def score(split: str) -> Tuple[int, int]:
            shortage = target_binary[split][majority_label] - split_bin_counts[split][majority_label]
            return (-shortage, split_total[split])  # 更缺少该类、总量更少优先

        best_split = min(["train", "val", "test"], key=score)
        split_data[best_split].extend(g_samples)
        split_total[best_split] += len(g_samples)
        split_bin_counts[best_split].update(g_bin)

    # 对未分组样本做分层切分（按 binary+type）
    random.Random(seed).shuffle(ungrouped)
    strata = defaultdict(list)
    for s in ungrouped:
        t_val = s.get("type_label", "Unknown")
        if isinstance(t_val, int):
            t_val = TYPE_LABELS.get(int(t_val), str(t_val))
        strata[(s["binary_label"], t_val)].append(s)

    for key, bucket in strata.items():
        key_seed = seed + (sum(ord(ch) for ch in str(key)) % 9973)
        rnd = random.Random(key_seed)
        rnd.shuffle(bucket)
        n = len(bucket)
        n_train = round(n * train_r)
        n_val = round(n * val_r)
        n_test = n - n_train - n_val
        split_data["train"].extend(bucket[:n_train])
        split_data["val"].extend(bucket[n_train:n_train + n_val])
        split_data["test"].extend(bucket[n_train + n_val:])

    return split_data["train"], split_data["val"], split_data["test"]


def compute_stats(name: str, data: List[Dict]) -> None:
    total = len(data)
    bin_counter = Counter(s.get("binary_label", 0) for s in data)
    type_counter = Counter()
    for s in data:
        t_val = s.get("type_label", "Unknown")
        if isinstance(t_val, int):
            t_val = TYPE_LABELS.get(int(t_val), str(t_val))
        type_counter[t_val] += 1
    label_type_counter = Counter(s.get("label_type", "strong") for s in data)

    safe = bin_counter.get(0, 0)
    mal = bin_counter.get(1, 0)
    print(f"{name:<6} total={total:5d}  safe={safe:5d}  malicious={mal:5d}")
    if total:
        print("  type distribution:")
        for tname, cnt in type_counter.items():
            print(f"    {str(tname):<16} {cnt:5d} ({cnt/total*100:5.1f}%)")
        print("  label_type distribution:")
        for lt, cnt in label_type_counter.items():
            print(f"    {lt:<6} {cnt:5d} ({cnt/total*100:5.1f}%)")


def save_json(data: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ 写出 {path.name}: {len(data)} 条")


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 比例之和需为 1.0，当前为 {total_ratio}")

    raw = load_dataset(input_path)
    data = [normalize_sample(s) for s in raw]

    train, val, test = split_with_groups(
        data,
        train_r=args.train_ratio,
        val_r=args.val_ratio,
        test_r=args.test_ratio,
        seed=args.seed,
    )

    print("-" * 80)
    compute_stats("train", train)
    compute_stats("val", val)
    compute_stats("test", test)

    save_json(train, output_dir / "train.json")
    save_json(val, output_dir / "val.json")
    save_json(test, output_dir / "test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线数据集切分")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "combined_dataset.json"))
    parser.add_argument("--output", type=str, default=str(DATA_DIR))
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    main(parser.parse_args())
