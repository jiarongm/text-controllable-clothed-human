r"""
measure_swap.py —— 量測資產切換（換裝／換體型）的延遲

背景
====
論文於 1.2.1、3.7.3 與 5.1 三處宣稱「資產切換的延遲為毫秒等級」，惟第四章
並無對應量測。本腳本補上該數據。

量測範圍
========
資產切換所需的工作為：自磁碟讀入 geometry.npz、skin_weights.npz 與
texture.png，將貼圖上傳至 GPU，再以原子性替換當前資產。**不含**生成，
生成屬離線階段（表 4-21）。

冷熱快取分別報告
================
首次讀取須實際存取磁碟，其後作業系統會將檔案留在頁快取中。二者可差一個
數量級，而使用者實際遇到的情形取決於該資產是否剛被存取過：預先生成的資產庫
於系統啟動後首次載入為冷、重複切換為熱。**僅報告熱快取會低估實際延遲**，
故本腳本兩者皆測。

用法
====
  python measure_swap.py --work demo_work/A2_navy_coat
  python measure_swap.py --work demo_work/A2_navy_coat --repeat 50

環境：chupa（需 torch）
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def drop_cache_hint():
    """嘗試清空頁快取。需 root，失敗則跳過並提示。"""
    try:
        subprocess.run(["sync"], check=True)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except Exception:
        return False


def load_once(work, to_gpu=True):
    """執行一次完整的資產載入，回傳各階段耗時（毫秒）。"""
    import cv2
    import torch

    t = {}
    w = Path(work)

    t0 = time.perf_counter()
    g = np.load(w / "geometry.npz")
    verts, faces, uv = g["verts"], g["faces"], g["uv"]
    t["geometry"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    sw = np.load(w / "skin_weights.npz")
    _j, _w = sw["joints"], sw["weights"]
    t["weights"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    tex = cv2.cvtColor(cv2.imread(str(w / "texture.png")), cv2.COLOR_BGR2RGB)
    t["texture_decode"] = (time.perf_counter() - t0) * 1e3

    if to_gpu and torch.cuda.is_available():
        t0 = time.perf_counter()
        tt = torch.tensor(tex.astype(np.float32) / 255.0, device="cuda")
        vv = torch.tensor(verts.astype(np.float32), device="cuda")
        torch.cuda.synchronize()      # 不同步會量到非同步排入佇列的時間
        t["gpu_upload"] = (time.perf_counter() - t0) * 1e3
        del tt, vv
    else:
        t["gpu_upload"] = float("nan")

    t["total"] = sum(v for v in t.values() if not np.isnan(v))
    return t


def summarise(rows, label):
    keys = ["geometry", "weights", "texture_decode", "gpu_upload", "total"]
    print(f"\n{label}（n = {len(rows)}）")
    print(f"  {'階段':16s} {'平均':>9s} {'標準差':>9s} {'P95':>9s} {'最大':>9s}")
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        if np.all(np.isnan(v)):
            continue
        star = "**" if k == "total" else "  "
        print(f"{star}{k:16s} {v.mean():8.2f} {v.std(ddof=1):9.2f} "
              f"{np.percentile(v, 95):9.2f} {v.max():9.2f}   ms")


def main():
    ap = argparse.ArgumentParser(description="量測資產切換延遲")
    ap.add_argument("--work", default="demo_work/A2_navy_coat")
    ap.add_argument("--repeat", type=int, default=30, help="熱快取重複次數")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    w = Path(args.work)
    for f in ("geometry.npz", "skin_weights.npz", "texture.png"):
        if not (w / f).exists():
            sys.exit(f"[錯誤] {w} 缺少 {f}")

    total_mb = sum((w / f).stat().st_size for f in
                   ("geometry.npz", "skin_weights.npz", "texture.png")) / 1e6
    print(f"資產：{w}　合計 {total_mb:.1f} MB")

    # 冷快取：僅一次，且需權限；失敗時明確說明而非靜默略過
    if drop_cache_hint():
        cold = [load_once(w)]
        summarise(cold, "冷快取（首次自磁碟讀取）")
    else:
        print("\n[略過] 冷快取量測需 root 權限清空頁快取。"
              "若需該數據，請以 sudo 執行；否則下列僅為熱快取結果，"
              "**會低估系統啟動後首次載入的延遲**。")

    for _ in range(args.warmup):
        load_once(w)
    hot = [load_once(w) for _ in range(args.repeat)]
    summarise(hot, "熱快取（重複切換）")

    tot = np.array([r["total"] for r in hot])
    print(f"\n判讀：熱快取下之切換延遲中位數 {np.median(tot):.1f} ms。")
    if np.median(tot) < 100:
        print("  → 支持論文「毫秒等級」之敘述（1.2.1、3.7.3、5.1 節）。")
    else:
        print("  → **不支持「毫秒等級」之敘述**，論文相關措辭須修正。")


if __name__ == "__main__":
    main()
