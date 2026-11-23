"""
export_betas_csv.py

把體型資產的 .pkl 裡「自己擬合出來的 10 維 betas」抽出成純數值 CSV。

為什麼需要這一步：
    rp_lib_*.pkl 是從 RenderPeople 的 emily 模板複製而來，裡面除了 betas
    還含 RenderPeople 的 A-pose、scale、translation，屬於該資料集的衍生資料，
    不可發布。但 betas 這 10 個數字是你自己由身高體重擬合出來的計算結果，
    與 RenderPeople 無關，可以發布。

    發布 CSV 而非 pkl，別人取得 RenderPeople 模板後，
    用 make_subject_from_measurements.py 就能重建出同樣的 subject。

在 chupa 環境、chupa 專案根目錄執行：

    python export_betas_csv.py
    python export_betas_csv.py --out data/results/shape_library_betas.csv

輸出：shape_library_betas.csv
"""
import argparse
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np

DEFAULT_OUT = "shape_library_betas.csv"
SEARCH_PATTERNS = [
    "data/renderpeople/smplx/rp_lib_*.pkl",
    "data/renderpeople/smplx/rp_user_*.pkl",
]


def extract_betas(path):
    """從 subject pkl 取出 betas，回傳 1 維 numpy 陣列或 None。"""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"  [略過] {os.path.basename(path)} 讀取失敗：{e}")
        return None

    if not isinstance(d, dict):
        print(f"  [略過] {os.path.basename(path)} 不是 dict")
        return None

    for key in ("betas", "shape", "shape_params"):
        if key in d:
            b = np.asarray(d[key], dtype=np.float64).reshape(-1)
            return b

    print(f"  [略過] {os.path.basename(path)} 找不到 betas（鍵：{list(d.keys())[:8]}）")
    return None


def parse_hw(name):
    """從 rp_lib_<h>_<w>_001 解析身高體重。"""
    m = re.match(r"rp_(?:lib|user)_(\d+)_(\d+)_", name)
    return (m.group(1), m.group(2)) if m else ("", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-betas", type=int, default=10,
                    help="輸出前幾維 betas（預設 10）")
    args = ap.parse_args()

    paths = []
    for pat in SEARCH_PATTERNS:
        paths.extend(glob.glob(pat))
    paths = sorted(set(paths))

    if not paths:
        sys.exit("[錯誤] 找不到任何 rp_lib_*.pkl / rp_user_*.pkl，"
                 "請確認在 chupa 專案根目錄執行")

    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        b = extract_betas(p)
        if b is None:
            continue
        n = min(args.n_betas, len(b))
        h, w = parse_hw(name)
        row = {"subject": name, "height_cm": h, "weight_kg": w}
        for i in range(args.n_betas):
            row[f"beta{i}"] = f"{b[i]:.6f}" if i < n else ""
        rows.append(row)
        print(f"  [取出] {name:<22} 身高 {h or '?':>3} 體重 {w or '?':>3}  "
              f"betas[0:3] = {np.round(b[:3], 4)}")

    if not rows:
        sys.exit("[錯誤] 沒有成功取出任何 betas")

    fields = ["subject", "height_cm", "weight_kg"] + \
             [f"beta{i}" for i in range(args.n_betas)]
    out = os.path.expanduser(args.out)
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n[完成] {len(rows)} 筆 → {os.path.abspath(out)}")
    print("這份 CSV 只含你自己擬合出的數值，可以發布；原始 .pkl 不要上傳。")


if __name__ == "__main__":
    main()
