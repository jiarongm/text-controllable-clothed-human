r"""
check_betas.py —— 統計九組體型之 SMPL-X 形狀參數，補齊 4.4.2 節的分量標準差

背景
====
論文 4.4.2 節原僅列出四個標準差（0.69、0.75、0.55、0.49），惟 β 為 10 維，
「各分量」卻只有四個數字，讀者無從得知其餘六維的情形。本腳本直接由九組
體型的擬合結果計算全部十個分量的統計量。

本腳本另檢驗 4.4.2 節的第二項敘述：β₀ 與身高、β₁ 與胖瘦是否確實呈單調
關聯。**該敘述為本研究於所測九組上的觀察，非 SMPL-X 的定義性質**，故須
以資料確認而不可視為當然。

用法
====
  python check_betas.py
  python check_betas.py --dir data/renderpeople/smplx --pattern "rp_demo_h*.pkl"

環境：chupa（需 numpy）
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np


def parse_hw(name):
    """由檔名取出身高與體重，例如 rp_demo_h165_w60_001 → (165, 60)。"""
    m = re.search(r"h(\d+)_w(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def monotone(pairs):
    """檢查 (x, y) 是否隨 x 單調。回傳 '遞增' / '遞減' / '非單調'。"""
    ys = [y for _, y in sorted(pairs)]
    if all(b >= a for a, b in zip(ys, ys[1:])):
        return "遞增"
    if all(b <= a for a, b in zip(ys, ys[1:])):
        return "遞減"
    return "**非單調**"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/renderpeople/smplx")
    ap.add_argument("--pattern", default="rp_demo_h*.pkl")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob(args.pattern))
    if not files:
        raise SystemExit(f"[錯誤] {args.dir} 下找不到符合 {args.pattern} 的檔案")

    rows = []
    for f in files:
        with open(f, "rb") as fh:
            p = pickle.load(fh, encoding="latin1")
        b = np.asarray(p["betas"], np.float64).reshape(-1)
        h, w = parse_hw(f.stem)
        rows.append({"name": f.stem, "h": h, "w": w, "beta": b})

    n_dim = min(len(r["beta"]) for r in rows)
    B = np.stack([r["beta"][:n_dim] for r in rows])
    print(f"體型組數 {len(rows)}，β 維度 {n_dim}\n")

    print("各組之身高／體重與前四個分量：")
    print(f"  {'名稱':28s} {'身高':>5s} {'體重':>5s} " +
          " ".join(f"β{i}".rjust(7) for i in range(min(4, n_dim))))
    for r in sorted(rows, key=lambda d: (d["h"] or 0, d["w"] or 0)):
        print(f"  {r['name']:28s} {str(r['h']):>5s} {str(r['w']):>5s} " +
              " ".join(f"{v:7.3f}" for v in r["beta"][:4]))

    print("\n全部分量之統計量（**此即論文表 4-9 所需**）：")
    print(f"  {'分量':>6s} {'平均':>8s} {'標準差':>8s} {'最小':>8s} {'最大':>8s}")
    for i in range(n_dim):
        c = B[:, i]
        print(f"  β{i:<5d} {c.mean():8.3f} {c.std(ddof=1):8.3f} "
              f"{c.min():8.3f} {c.max():8.3f}")

    # 標準差若於高階分量仍不可忽略，則「僅前四維有意義」的說法不成立
    sd = B.std(axis=0, ddof=1)
    print(f"\n  前四維標準差合計 {sd[:4].sum():.3f}，"
          f"第五維以後合計 {sd[4:].sum():.3f}"
          f"（佔全部 {100*sd[4:].sum()/sd.sum():.1f}%）")
    if sd[4:].sum() / sd.sum() > 0.25:
        print("  → 高階分量的變異**不可忽略**，論文不宜只報告前四維，"
              "應完整列出十維。")
    else:
        print("  → 變異主要集中於前四維，惟論文仍應完整列出十維以免讀者疑義。")

    print("\n4.4.2 節之單調性敘述核對：")
    hs = [(r["h"], r["beta"][0]) for r in rows if r["h"] is not None]
    ws = [(r["w"] / (r["h"] / 100.0) ** 2, r["beta"][1])
          for r in rows if r["h"] and r["w"]]
    print(f"  β0 隨身高：{monotone(hs)}")
    print(f"  β1 隨 BMI：{monotone(ws)}")
    print("  （非單調者，4.4.2 節之敘述須改寫；BMI 以 體重/身高² 計算）")


if __name__ == "__main__":
    main()
