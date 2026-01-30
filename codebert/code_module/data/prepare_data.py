"""
Offline data preparation: merge local sources into a unified dataset with fields:
- id: unique integer
- code: code text
- binary_label: 0/1 (0=normal, 1=malicious)
- type_label: int (0=Normal, >0 malicious type, -1 unknown malicious type)
- label_type: "strong" | "weak"
- source: provenance string

Sources considered (all local, no network):
1) 旧/数据提取/final_dataset.json  (primary strong labels)
2) 旧/数据提取/漏洞代码/webshell.json (assumed malicious, type=WebShell, strong)
3) 旧/数据提取/漏洞代码/raw_samples.json (assumed malicious, type unknown, weak)
4) 旧/数据提取/(安全)raw_samples/ directory (assumed safe, strong)

Output:
    code_module/data/combined_dataset.json

Run from repo root:
    python code_module/data/prepare_data.py
"""
from pathlib import Path
import json
from typing import Dict, List
import sys

# Ensure repo root (contains common/) is on sys.path for flexible CWD execution
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.config import BASE_DIR, DATA_DIR, TYPE_TO_ID, TYPE_LABELS
from common.utils import save_json

RAW_BASE = BASE_DIR / "旧" / "数据提取"
VULN_DIR = RAW_BASE / "漏洞代码"
SAFE_DIR = RAW_BASE / "(安全)raw_samples"
OUTPUT_PATH = DATA_DIR / "combined_dataset.json"


def load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"[MISS] {path}")
        return []
    try:
        data = json.load(path.open("r", encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return []
    if not isinstance(data, list):
        print(f"[SKIP] {path}: top-level is {type(data)}")
        return []
    return data


def infer_type_label(malicious_type_field) -> int:
    if malicious_type_field is None:
        return -1
    if isinstance(malicious_type_field, str):
        types = [malicious_type_field]
    elif isinstance(malicious_type_field, list):
        types = malicious_type_field
    else:
        return -1
    for t in types:
        t_low = str(t).lower()
        if "webshell" in t_low or "shell" in t_low:
            return TYPE_TO_ID.get("WebShell", 4)
        if "command" in t_low or "exec" in t_low:
            return TYPE_TO_ID.get("CommandInjection", 4)
        if "backdoor" in t_low or "remote" in t_low or "rce" in t_low:
            return TYPE_TO_ID.get("Backdoor", 4)
    return TYPE_TO_ID.get("OtherMalicious", 4)


def from_primary() -> List[Dict]:
    primary_path = RAW_BASE / "final_dataset.json"
    fallback_path = RAW_BASE / "labeled_dataset.json"
    path = primary_path if primary_path.exists() else fallback_path
    data = load_json_list(path)
    result = []
    for s in data:
        code = s.get("code")
        label = s.get("label")
        if not code or not isinstance(code, str):
            continue
        if label is None:
            continue
        try:
            bi = int(label)
        except Exception:
            continue
        if bi not in (0, 1):
            continue
        type_label = infer_type_label(s.get("malicious_type")) if bi == 1 else 0
        result.append({
            "code": code,
            "binary_label": bi,
            "type_label": type_label,
            "label_type": "strong",
            "source": s.get("source", "final_dataset"),
        })
    print(f"[MAIN] primary dataset kept: {len(result)}")
    return result


def from_webshell() -> List[Dict]:
    data = load_json_list(VULN_DIR / "webshell.json")
    result = []
    for idx, s in enumerate(data):
        code = s.get("code")
        if not code or not isinstance(code, str):
            continue
        result.append({
            "code": code,
            "binary_label": 1,
            "type_label": TYPE_TO_ID.get("WebShell", 4),
            "label_type": "strong",
            "source": s.get("source", f"vuln/webshell[{idx}]"),
        })
    print(f"[WEB] webshell added: {len(result)}")
    return result


def from_vuln_raw() -> List[Dict]:
    data = load_json_list(VULN_DIR / "raw_samples.json")
    result = []
    for idx, s in enumerate(data):
        code = s.get("code")
        if not code or not isinstance(code, str):
            continue
        result.append({
            "code": code,
            "binary_label": 1,
            "type_label": -1,  # unknown malicious type
            "label_type": "weak",  # 视为弱标注
            "source": s.get("source", f"vuln/raw[{idx}]"),
        })
    print(f"[VULN] raw_samples added (weak): {len(result)}")
    return result


def from_safe_raw() -> List[Dict]:
    result: List[Dict] = []
    if not SAFE_DIR.exists():
        print(f"[MISS] safe raw dir: {SAFE_DIR}")
        return result
    for path in SAFE_DIR.rglob("*"):
        if path.is_dir():
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not code:
            continue
        result.append({
            "code": code,
            "binary_label": 0,
            "type_label": 0,
            "label_type": "strong",
            "source": str(path.relative_to(BASE_DIR)),
        })
    print(f"[SAFE] safe raw files added: {len(result)}")
    return result


def main():
    all_samples: List[Dict] = []
    all_samples.extend(from_primary())
    all_samples.extend(from_webshell())
    all_samples.extend(from_vuln_raw())
    all_samples.extend(from_safe_raw())

    # deduplicate by code text
    deduped: List[Dict] = []
    seen = set()
    for s in all_samples:
        code = s["code"]
        if code in seen:
            continue
        seen.add(code)
        deduped.append(s)

    # assign ids
    for i, s in enumerate(deduped):
        s["id"] = i

    pos = sum(1 for s in deduped if s["binary_label"] == 1)
    neg = sum(1 for s in deduped if s["binary_label"] == 0)
    print("-" * 80)
    print(f"合并后样本数: {len(deduped)} (pos={pos}, neg={neg})")

    save_json(deduped, OUTPUT_PATH)
    print(f"✅ 写出 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
