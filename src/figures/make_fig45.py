r"""
make_fig45.py —— 產生論文圖 4-5（不同服裝提示的生成結果）

圖 4-5 需要四張**正面法向量圖**並排：貼身上衣長褲、短袖短褲、連身長裙、
兩件式組合（外套與裙）。

圖說寫明四者「皆以同一 A-pose 人體先驗、相同生成參數（停用兩處重採樣）
生成」。**此條件必須逐項核對，不可僅憑目視挑選**——若某一類的引導強度或
採樣步數與其餘不同，該圖即不構成公平比較，而此種不一致無法由圖面看出。
本腳本因此於併圖前列出各類的實際參數，並於發現不一致時明確警告。

用法
====
  # 先看有哪些候選（不併圖），確認四類是否齊備
  python make_fig45.py --list

  # 併圖
  python make_fig45.py --out figures

  # 換對象或放寬條件
  python make_fig45.py --subject rp_emily_apose_001 --allow-mismatch

環境：chupa（需 opencv、numpy）
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "renderpeople"

# (圖上標籤, 目錄名需含的關鍵字)。順序即圖 4-5 由左至右的順序。
#
# 關鍵字取自實際的目錄名，勿以提示詞的中文說法反推——目錄名使用連字號
# （t-shirt）且第四類的實際用字為 jacket 而非 coat，二者皆曾導致比對落空。
CATEGORIES = [
    ("sleeveless top + long pants", ["sleeveless", "long_pants"]),
    ("t-shirt + shorts",            ["t-shirt", "shorts"]),
    ("long dress",                  ["long_dress"]),
    ("jacket + skirt",              ["jacket", "skirt"]),
]

CELL = 512


def read_cfg(p):
    if not p.exists():
        return {}
    t = p.read_text(encoding="utf-8", errors="ignore")

    def g(k, cast=int):
        m = re.search(rf"^\s*{k}\s*:\s*([\d.]+)", t, re.M)
        return cast(m.group(1)) if m else None
    return {"text_rs": g("n_resample_from_text"),
            "refine_rs": g("n_resample"),
            "st": g("text_cfg_scale", float),
            "steps": g("text_steps")}


def candidates(subject):
    """回傳每一類的候選執行，已依時間新到舊排序。"""
    out = {label: [] for label, _ in CATEGORIES}
    if not RESULTS.is_dir():
        sys.exit(f"[錯誤] 找不到 {RESULTS}")
    for nf in RESULTS.glob("*/*/*/normal_F/000.png"):
        run = nf.parent.parent
        if subject and run.parent.name != subject:
            continue
        tag = run.name.lower()
        cfg = read_cfg(run.parent.parent / "config.yaml")
        for label, keys in CATEGORIES:
            if all(k in tag for k in keys):
                out[label].append({"png": nf, "tag": run.name,
                                   "stamp": run.parent.parent.name, **cfg})
    for v in out.values():
        v.sort(key=lambda d: d["stamp"], reverse=True)
    return out


def clean(rows):
    """僅保留兩處重採樣皆停用者——此為圖說所載明的條件。"""
    return [r for r in rows if r.get("text_rs") == 0 and r.get("refine_rs") == 0]


def show(cands):
    for label, _ in CATEGORIES:
        rows = cands[label]
        ok = clean(rows)
        print(f"\n【{label}】候選 {len(rows)} 筆，其中兩處重採樣皆停用者 {len(ok)} 筆")
        for r in rows[:6]:
            mark = "  ✓" if r in ok else "   "
            print(f"  {mark} {r['stamp']}  st={r.get('st')}  "
                  f"steps={r.get('steps')}  文字rs={r.get('text_rs')}  "
                  f"精修rs={r.get('refine_rs')}")
            print(f"      {r['tag']}")
        if not ok:
            print("      **無符合條件者**——該類無法納入圖 4-5，"
                  "或須放寬條件並於圖說中據實說明。")


def load_cell(png):
    import cv2
    img = cv2.imread(str(png))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-5")
    ap.add_argument("--subject", default="rp_emily_apose_001")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="即使生成參數不一致仍併圖（將於圖說中標註）")
    ap.add_argument("--cols", type=int, default=0,
                    help="每列格數；預設四類時為 2（2×2），其餘為單列")
    args = ap.parse_args()

    cands = candidates(args.subject)
    if args.list:
        show(cands)
        return

    import cv2
    picked, missing = [], []
    for label, _ in CATEGORIES:
        ok = clean(cands[label])
        if not ok:
            missing.append(label)
        else:
            picked.append((label, ok[0]))

    if missing:
        print(f"[錯誤] 以下類別無符合條件的執行：{'、'.join(missing)}")
        print("  請執行 python make_fig45.py --list 檢視候選，"
              "或補跑該類的生成。")
        if not args.allow_mismatch:
            sys.exit(1)

    # 參數一致性檢核——圖說宣稱「相同生成參數」，此處逐項確認
    print("\n各類所選之執行與其生成參數：")
    for label, r in picked:
        print(f"  {label:26s} {r['stamp']}  st={r['st']}  steps={r['steps']}")
    sts = {r["st"] for _, r in picked}
    sps = {r["steps"] for _, r in picked}
    consistent = len(sts) == 1 and len(sps) == 1
    if consistent:
        print(f"\n  ✓ 四者的引導強度與採樣步數一致"
              f"（st={sts.pop()}、steps={sps.pop()}），符合圖說之宣稱。")
    else:
        print(f"\n  ✗ **生成參數不一致**：st={sorted(sts)}、steps={sorted(sps)}")
        print("     圖說「相同生成參數」之敘述與事實不符，"
              "須改用參數一致的執行，或修正圖說。")
        if not args.allow_mismatch:
            sys.exit(1)

    cells, labels = [], []
    for label, r in picked:
        c = load_cell(r["png"])
        if c is None:
            print(f"[警告] 無法讀取 {r['png']}")
            continue
        cells.append(c)
        labels.append(label)

    band = 36
    tiles = []
    for c, t in zip(cells, labels):
        b = np.full((band, CELL, 3), 255, np.uint8)
        cv2.putText(b, t, (8, band - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([c, b]))

    # 2×2 排列。四格橫排時每格於頁寬下僅約 4 公分，袖長、褲長、裙襬等
    # 判定用的特徵會縮到難以辨認；2×2 可使每格寬度加倍。
    if args.cols and args.cols > 0:
        ncol = args.cols
    else:
        ncol = 2 if len(tiles) == 4 else len(tiles)

    rows = [tiles[i:i + ncol] for i in range(0, len(tiles), ncol)]
    # 末列不足時以白格補滿，避免 hstack 因寬度不一而失敗
    h, w = tiles[0].shape[:2]
    for r in rows:
        while len(r) < ncol:
            r.append(np.full((h, w, 3), 255, np.uint8))
    grid = np.vstack([np.hstack(r) for r in rows])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_4-5_prompts.png"
    cv2.imwrite(str(p), grid)
    print(f"\n[完成] {p}（{len(cells)} 類，{len(rows)} 列 × {ncol} 行）")
    if not consistent:
        print("  **本圖之生成參數不一致，圖說須據實修正。**")


if __name__ == "__main__":
    main()
