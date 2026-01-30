"""
构建统一数据集脚本：
- 扫描多处目录的 json/jsonl/csv 代码数据
- 自动探测文件格式，尝试映射 code/binary/type/label_type 等字段
- 去重并按强标注/弱标注/未标注分类输出
运行：python tools/build_unified_datasets.py
"""
import os
import json
import csv
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Iterable

# 待扫描的目录（按优先级）
SCAN_DIRS = [
    Path("code_module/data"),
    Path("data"),
    Path("datasets"),
    Path("raw_data"),
    Path("旧"),
    Path("old"),
]

# 输出目录
OUTPUT_DIR = Path("code_module/data/unified")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 输出文件名
OUT_STRONG_BT = OUTPUT_DIR / "strong_labeled_binary_and_type.json"
OUT_STRONG_B = OUTPUT_DIR / "strong_labeled_binary_only.json"
OUT_WEAK = OUTPUT_DIR / "weak_labeled.json"
OUT_UNLABELED = OUTPUT_DIR / "unlabeled_pool.json"
OUT_CONFLICT = OUTPUT_DIR / "conflicted_samples.json"

# 候选字段名映射
CODE_FIELDS = ["code", "source", "snippet", "content", "body", "func", "text"]
BINARY_FIELDS = ["binary_label", "label", "is_vul", "is_malicious", "malicious", "target", "vulnerability", "vul"]
TYPE_FIELDS = ["type_label", "type", "vul_type", "category", "cwe", "cwe_id", "vuln_type"]
QUALITY_FIELDS = ["label_type", "is_weak", "weak", "source_type", "label_quality"]
ID_FIELDS = ["id", "sample_id", "uid"]
SOURCE_FIELDS = ["source", "file_path", "path", "repo", "dataset"]

MIN_CODE_LEN = 5  # 小于此长度视为无效


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def load_file(path: Path) -> Iterable[Dict[str, Any]]:
    """根据扩展和内容类型加载文件，返回样本迭代器。"""
    if path.suffix.lower() in {".json"}:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return []
        if txt.lstrip().startswith("["):
            try:
                data = json.loads(txt)
                return data if isinstance(data, list) else []
            except Exception:
                print(f"[warn] 跳过无法解析的 JSON 列表文件: {path}")
                return []
        else:
            # 可能是 jsonl 但扩展名为 json
            rows = []
            for line in txt.splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    print(f"[warn] 跳过无法解析的 JSON 行: {path}")
                    return []
            return rows
    if path.suffix.lower() in {".jsonl", ".jl"}:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    if path.suffix.lower() in {".csv"}:
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    # 其他扩展暂不解析
    return []


def pick_first(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def to_int_label(val):
    if val is None:
        return None
    # bool
    if isinstance(val, bool):
        return 1 if val else 0
    # numeric
    if isinstance(val, (int, float)):
        return int(val)
    # string
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "malicious", "vul", "vulnerable"}:
        return 1
    if s in {"0", "false", "no", "n", "benign", "safe", "normal"}:
        return 0
    try:
        return int(s)
    except Exception:
        return None


def detect_quality(sample: Dict[str, Any]) -> str:
    """根据字段推断标注质量 strong/weak/none。"""
    # 显式弱标注
    weak_flags = {"weak", "model", "pseudo", "auto", "soft"}
    for k in QUALITY_FIELDS:
        if k in sample:
            v = sample[k]
            if isinstance(v, str):
                vs = v.lower()
                if vs in weak_flags:
                    return "weak"
                if vs in {"strong", "human", "manual"}:
                    return "strong"
            if isinstance(v, bool):
                if v:
                    return "weak"
    # 默认 strong 如果有标签，否则 none
    has_label = (pick_first(sample, BINARY_FIELDS) is not None) or (pick_first(sample, TYPE_FIELDS) is not None)
    return "strong" if has_label else "none"


def normalize_sample(raw: Dict[str, Any], source_name: str, id_prefix: str, idx: int) -> Dict[str, Any]:
    code = pick_first(raw, CODE_FIELDS)
    if not code or len(str(code).strip()) < MIN_CODE_LEN:
        return {}
    binary_raw = pick_first(raw, BINARY_FIELDS)
    type_raw = pick_first(raw, TYPE_FIELDS)
    binary_label = to_int_label(binary_raw)
    # type_label 尝试转 int，否则置 None
    type_label = None
    if type_raw is not None:
        try:
            type_label = int(type_raw)
        except Exception:
            type_label = None
    quality = detect_quality(raw)
    src = pick_first(raw, SOURCE_FIELDS) or source_name
    rid = pick_first(raw, ID_FIELDS)
    if rid is None:
        rid = f"{id_prefix}_{idx}"
    return {
        "id": str(rid),
        "code": str(code),
        "binary_label": binary_label,
        "type_label": type_label,
        "label_quality": quality if quality in {"strong", "weak"} else ("none" if (binary_label is None and type_label is None) else "strong"),
        "source": str(src),
    }


def scan_files() -> List[Path]:
    files = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for ext in ("*.json", "*.jsonl", "*.jl", "*.csv"):
            files.extend(base.rglob(ext))
    # 排除输出目录自身
    files = [p for p in files if OUTPUT_DIR not in p.parents]
    return files


def merge_and_dedup(samples: List[Dict[str, Any]]):
    by_hash = {}
    conflicts = []
    for s in samples:
        h = md5_text(s["code"])
        existing = by_hash.get(h)
        if existing is None:
            by_hash[h] = s
            continue
        # 冲突处理：优先 strong > weak；标签冲突记入 conflicts
        def weight(q):
            return {"strong": 2, "weak": 1, "none": 0}.get(q, 0)
        new_w = weight(s.get("label_quality", "none"))
        old_w = weight(existing.get("label_quality", "none"))
        # 检查标签冲突
        if (existing.get("binary_label") is not None and s.get("binary_label") is not None and existing["binary_label"] != s["binary_label"]):
            conflicts.append({"existing": existing, "new": s, "reason": "binary_conflict", "hash": h})
            # 选择权重高者
            if new_w > old_w:
                by_hash[h] = s
            continue
        if (existing.get("type_label") is not None and s.get("type_label") is not None and existing["type_label"] != s["type_label"]):
            conflicts.append({"existing": existing, "new": s, "reason": "type_conflict", "hash": h})
            if new_w > old_w:
                by_hash[h] = s
            continue
        # 无冲突，选择权重高者
        if new_w > old_w:
            by_hash[h] = s
    return list(by_hash.values()), conflicts


def split_buckets(samples: List[Dict[str, Any]]):
    strong_bt, strong_b, weak, unlabeled = [], [], [], []
    for s in samples:
        q = s.get("label_quality", "none")
        b = s.get("binary_label")
        t = s.get("type_label")
        if b is None and t is None:
            unlabeled.append(s)
            continue
        if q == "weak":
            weak.append(s)
            continue
        # 强标注
        if b in (0, 1) and t is not None:
            strong_bt.append(s)
        elif b in (0, 1):
            strong_b.append(s)
        else:
            # 有类型但无 binary，放入 weak 以防止污染主训（也可放 unlabeled，取保守策略）
            weak.append(s)
    return strong_bt, strong_b, weak, unlabeled


def save_json(path: Path, data: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize(name: str, data: List[Dict[str, Any]]):
    from collections import Counter
    b_cnt = Counter()
    t_cnt = Counter()
    q_cnt = Counter()
    for s in data:
        b_cnt[s.get("binary_label")] += 1
        t_cnt[s.get("type_label")] += 1
        q_cnt[s.get("label_quality")] += 1
    print(f"[{name}] samples={len(data)} binary_dist={dict(b_cnt)} type_dist={dict(t_cnt)} quality_dist={dict(q_cnt)}")


def main():
    files = scan_files()
    print(f"发现候选数据文件 {len(files)} 个")
    all_raw = []
    for p in files:
        samples = list(load_file(p))
        if not samples:
            continue
        print(f"载入 {p} 样本数 {len(samples)}")
        id_prefix = p.stem
        for idx, r in enumerate(samples):
            norm = normalize_sample(r, source_name=str(p), id_prefix=id_prefix, idx=idx)
            if norm:
                all_raw.append(norm)
    print(f"总计规范化样本 {len(all_raw)}")

    deduped, conflicts = merge_and_dedup(all_raw)
    print(f"去重后样本 {len(deduped)}，冲突 {len(conflicts)}")

    strong_bt, strong_b, weak, unlabeled = split_buckets(deduped)

    save_json(OUT_STRONG_BT, strong_bt)
    save_json(OUT_STRONG_B, strong_b)
    save_json(OUT_WEAK, weak)
    save_json(OUT_UNLABELED, unlabeled)
    save_json(OUT_CONFLICT, conflicts)

    summarize("strong_binary_and_type", strong_bt)
    summarize("strong_binary_only", strong_b)
    summarize("weak", weak)
    summarize("unlabeled", unlabeled)
    summarize("conflicts", conflicts)

    print("输出文件：")
    for path in [OUT_STRONG_BT, OUT_STRONG_B, OUT_WEAK, OUT_UNLABELED, OUT_CONFLICT]:
        print(f"  {path} -> {path.exists() and path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
