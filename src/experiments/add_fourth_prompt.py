r"""
add_fourth_prompt.py —— 為既有掃描補上第四個提示（兩件式組合）

背景
====
論文表 4-8 之三個提示（長裙、無袖長褲、T 恤短褲）各 27 次生成，合計 81 次。
第四類「外套＋裙」從未納入掃描，其「品質較差」之評語僅來自四個樣本的目視
印象，且圖 4-5 第四例並未顯現該現象。本腳本補齊該類的量化資料，使其與
其餘三類於相同條件下可直接比較。

為何不直接改 run_factor_sweep.py 重跑
=====================================
該腳本的 `--stage sheet` 於 `judgement.csv` 已存在時不覆寫，加 `--force`
則會**清空既有的 117 筆人工判定**。本腳本因此自行處理 CSV，僅以附加方式
寫入新列，既有判定完全不動。

生成規模
========
13 個唯一條件 × 3 個種子 = 39 次生成，採第一階段早停（每次約 90 秒），
合計約一小時。判定圖僅含新提示，無須重看舊樣本。

用法
====
  python add_fourth_prompt.py --dry-run     # 先看將執行哪些指令
  python add_fourth_prompt.py               # 實際生成
  python add_fourth_prompt.py --sheet-only  # 生成已完成，只補 CSV 與判定圖

環境：chupa
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from run_factor_sweep import (
        CHUPA_PROJECT_DIR, SEEDS, STAGE1_ONLY,
        enumerate_conditions, build_cmd, _find_normal_map,
        run_until_normal_map, PROMPTS,
    )
except ImportError as e:
    sys.exit(f"[錯誤] 需與 run_factor_sweep.py 置於同一目錄：{e}")

# 新提示。名稱須與既有三者同為簡短鍵值，供 CSV 與分析階段使用。
NEW_NAME = "jacket_skirt"
NEW_PROMPT = "a woman wearing a jacket and a skirt"

# 判定準則。須與 4.2.4 節同為**連言**準則——兩個服裝概念都要正確，
# 僅其一正確者不計分，方與其餘三類的嚴格程度一致。
#
# **本類另行分項記錄，而非僅記連言結果。** 其餘三類的兩項特徵同屬一件
# 服裝（如短袖與短褲雖分屬上下身，但「T 恤短褲」為單一套裝概念）；本類
# 則為兩個獨立的服裝概念（外套、裙），故失敗可能發生於任一邊。分項記錄
# 方能回答「失敗發生在哪一邊」——若外套幾乎總是正確而裙裝崩潰，即可與
# 長裙提示（裙裝單獨出現時成功率 81.5%）對照，支持 4.2.4 節所述之
# 相乘結構；僅記連言結果則無從分辨。
#
# 性別不列入判定：人體先驗固定，體型不由文字決定，其餘三類亦未判性別。
SUBCRITERIA = [
    ("correct_jacket", "上身是否為外套（可見門襟或前開襟）"),
    ("correct_skirt",  "下身是否為裙裝（兩腿間連續之布面，非分離的兩腿）"),
]
CRITERION = ("分兩欄判定，兩者皆為 1 時 correct 方為 1：\n"
             + "\n".join(f"    {k}　{d}" for k, d in SUBCRITERIA))


def finalize(out):
    r"""由兩個分項欄計算 correct，並報告失敗集中於哪一邊。

    `run_factor_sweep.py --stage analyze` 讀取的是 `correct` 欄，故分項
    判定完成後須據此回填。本函式同時輸出分項的通過率——該資訊為單一
    連言欄所無法提供，亦為 4.2.5 節論證所需。
    """
    csv_path = Path(out) / "judgement.csv"
    if not csv_path.exists():
        sys.exit(f"[錯誤] 找不到 {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header, rows = list(rd.fieldnames), list(rd)

    sub = [k for k, _ in SUBCRITERIA]
    miss = [c for c in sub if c not in header]
    if miss:
        sys.exit(f"[錯誤] judgement.csv 缺少欄位 {miss}；"
                 "請先執行 python add_fourth_prompt.py --sheet-only")

    target = [r for r in rows if r.get("prompt") == NEW_NAME]
    done = [r for r in target
            if all(r.get(c, "").strip() in ("0", "1") for c in sub)]
    if len(done) < len(target):
        print(f"[注意] {NEW_NAME} 共 {len(target)} 列，"
              f"其中 {len(target) - len(done)} 列尚未完成分項判定，將略過。")

    for r in done:
        vals = [int(r[c]) for c in sub]
        r["correct"] = "1" if all(v == 1 for v in vals) else "0"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    if not done:
        print("尚無可彙整的判定。")
        return

    def block(rows, title):
        if not rows:
            return None
        m = len(rows)
        print(f"\n{title}（{m} 次生成）")
        for c, desc in SUBCRITERIA:
            k = sum(1 for r in rows if r[c] == "1")
            print(f"  {c:16s} {k:3d}/{m:<3d} {100 * k / m:5.1f}%　{desc}")
        k = sum(1 for r in rows if r["correct"] == "1")
        print(f"  {'correct（連言）':16s} {k:3d}/{m:<3d} {100 * k / m:5.1f}%")
        return k / m

    print(f"\n{NEW_NAME}　已判定 {len(done)} 次生成")

    # **須分開報告。** 表 4-8 僅計重採樣為 0 的九個條件；重採樣 ≥ 1 的條件
    # 於其餘三類皆為 0%，若混入將使本類的比率被系統性拉低而不可比。
    rs0 = [r for r in done if str(r.get("resample", "")).strip() == "0"]
    rsn = [r for r in done if str(r.get("resample", "")).strip() not in ("0", "")]
    block(done, "全部條件")
    p0 = block(rs0, "**重採樣 = 0（此為與表 4-8 可比者）**")
    block(rsn, "重採樣 ≥ 1")

    if p0 is not None:
        print(f"\n  與表 4-8 並列：長裙 81.5%、無袖長褲 66.7%、"
              f"T 恤短褲 48.1%、外套裙 {100 * p0:.1f}%")
        if p0 >= 0.80:
            print("  → **本類並非最差，與長裙相當。**"
                  "表 4-9 之「較差」與 4.2.5 之相關敘述須據此修正；"
                  "4.2.4 的相乘結構觀察亦須重新檢視——"
                  "兩項偏離未必導致更低的正確率。")
        elif p0 <= 0.50:
            print("  → 本類為四者中最低，支持「兩件式組合較難」之敘述。")
        else:
            print("  → 本類落於中段，「兩件式組合最差」之敘述不成立，須改寫。")

    n = len(done)
    ja = sum(1 for r in done if r[sub[0]] == "1")
    sk = sum(1 for r in done if r[sub[1]] == "1")
    print()
    if ja - sk >= max(3, 0.2 * n):
        print("  → 失敗明顯集中於**裙裝**一側。可與長裙提示（裙裝單獨出現時"
              f"為 81.5%）對照——若本類的裙裝通過率遠低於該值，"
              "即支持「加入第二個服裝概念會拖累原本可達成的特徵」。")
    elif sk - ja >= max(3, 0.2 * n):
        print("  → 失敗明顯集中於**外套**一側，與預期相反，須於正文說明。")
    else:
        print("  → 兩側的失敗率相近，無法歸因於單一服裝概念。")
    print(f"\n接著執行：python run_factor_sweep.py --stage analyze --out {out}")


def main():
    ap = argparse.ArgumentParser(description="為既有掃描補上第四個提示")
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sheet-only", action="store_true",
                    help="跳過生成，只補 judgement.csv 與判定圖")
    ap.add_argument("--force", action="store_true",
                    help="既有 PNG 亦重新生成")
    ap.add_argument("--finalize", action="store_true",
                    help="分項判定完成後，據以回填 correct 欄並報告分項通過率")
    args = ap.parse_args()

    if args.finalize:
        finalize(args.out)
        return

    if NEW_NAME in dict(PROMPTS):
        print(f"[注意] run_factor_sweep.PROMPTS 已含 {NEW_NAME}；"
              "本腳本仍以自身的定義執行，兩者須一致。")

    out = Path(args.out)
    normals = out / "normals"
    normals.mkdir(parents=True, exist_ok=True)
    conds = enumerate_conditions()

    import cv2

    # ---------- 生成 ----------
    new_rows = []
    total = len(conds) * len(SEEDS)
    if not args.sheet_only:
        n, t0 = 0, time.time()
        for k, c in conds.items():
            for seed in SEEDS:
                n += 1
                tag = f"{k}__{NEW_NAME}__s{seed}"
                dst = normals / f"{tag}.png"
                rec = {"tag": tag, "cond": k, "prompt": NEW_NAME,
                       "seed": seed, "png": str(dst),
                       "st": c["st"], "steps": c["steps"],
                       "resample": c["resample"]}

                if dst.exists() and not args.force:
                    print(f"[{n}/{total}] {tag} 已存在，跳過")
                    new_rows.append(rec)
                    continue

                cmd = build_cmd(c, NEW_PROMPT, seed, stage1_only=True)
                print(f"[{n}/{total}] {tag}")
                if args.dry_run:
                    print("  $ " + " ".join(str(x) for x in cmd))
                    continue

                t = time.time()
                if STAGE1_ONLY:
                    r = subprocess.run(cmd, cwd=CHUPA_PROJECT_DIR,
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  ✗ 失敗：{r.stderr[-600:]}")
                        continue
                    src = _find_normal_map(t)
                else:
                    src = run_until_normal_map(cmd)

                if src is None:
                    print("  ✗ 找不到法向量圖輸出")
                    continue
                cv2.imwrite(str(dst), cv2.imread(str(src)))
                print(f"    {(time.time() - t):.0f} 秒")
                new_rows.append(rec)

        if args.dry_run:
            print(f"\n[dry-run] 共 {total} 次生成，未實際執行。")
            return
        print(f"\n生成完成 {len(new_rows)} 筆，"
              f"耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    else:
        for k, c in conds.items():
            for seed in SEEDS:
                tag = f"{k}__{NEW_NAME}__s{seed}"
                dst = normals / f"{tag}.png"
                if dst.exists():
                    new_rows.append({"tag": tag, "cond": k, "prompt": NEW_NAME,
                                     "seed": seed, "png": str(dst),
                                     "st": c["st"], "steps": c["steps"],
                                     "resample": c["resample"]})
        print(f"找到既有的 {len(new_rows)} 張法向量圖")

    if not new_rows:
        sys.exit("[錯誤] 沒有可用的生成結果")

    # ---------- 附加至 manifest.json ----------
    # **不可略過此步。** run_factor_sweep.py --stage analyze 依 manifest
    # 決定每個條件包含哪些樣本；若僅寫入 judgement.csv 而未更新 manifest，
    # 分析階段將完全看不到新提示，且會因「已判定筆數多於 manifest 筆數」
    # 而印出負數的未判定警告。
    man_path = out / "manifest.json"
    man = []
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
    have = {m["tag"] for m in man}
    conds_by_key = {k: c for k, c in conds.items()}
    added_man = 0
    for r in new_rows:
        if r["tag"] in have:
            continue
        c = conds_by_key[r["cond"]]
        man.append({"tag": r["tag"], "cond": r["cond"], "prompt": r["prompt"],
                    "seed": r["seed"], "png": r["png"], **c})
        added_man += 1
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"manifest.json：新增 {added_man} 筆，現共 {len(man)} 筆")

    # ---------- 附加至 judgement.csv，不動既有判定 ----------
    csv_path = out / "judgement.csv"
    base_cols = ["tag", "cond", "prompt", "seed", "st", "steps",
                 "resample", "correct"]
    sub_cols = [k for k, _ in SUBCRITERIA]
    existing, old_rows, header = set(), [], base_cols

    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            header = list(rd.fieldnames or base_cols)
            old_rows = list(rd)
            existing = {r["tag"] for r in old_rows}
        print(f"既有 judgement.csv 共 {len(existing)} 列")

    # 分項欄位為本類專有；其餘三類留空。加欄須重寫整份檔案，故先確認
    # 既有的 correct 值全數保留後再寫出。
    need_cols = [c for c in sub_cols if c not in header]
    if need_cols:
        header = header + need_cols
        print(f"新增欄位 {need_cols}——既有 {len(old_rows)} 列的判定值將原樣保留")

    new_csv = []
    for r in old_rows:
        new_csv.append({c: r.get(c, "") for c in header})
    added = 0
    for r in sorted(new_rows, key=lambda d: (d["cond"], d["seed"])):
        if r["tag"] in existing:
            continue
        row = {c: "" for c in header}
        row.update({"tag": r["tag"], "cond": r["cond"], "prompt": r["prompt"],
                    "seed": r["seed"], "st": r["st"], "steps": r["steps"],
                    "resample": r["resample"]})
        new_csv.append(row)
        added += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(new_csv)
    print(f"已附加 {added} 列（既有 {len(existing)} 列之判定值未更動）")

    # ---------- 判定圖：僅新提示，每條件一張（3 個種子並排） ----------
    jdir = out / f"judge_{NEW_NAME}"
    jdir.mkdir(parents=True, exist_ok=True)
    by_cond = {}
    for r in new_rows:
        by_cond.setdefault(r["cond"], []).append(r)

    for cond, items in by_cond.items():
        tiles = []
        for m in sorted(items, key=lambda d: d["seed"]):
            im = cv2.imread(m["png"])
            if im is None:
                continue
            im = cv2.resize(im, (380, 380))
            band = np.full((30, 380, 3), 255, np.uint8)
            cv2.putText(band, f"seed {m['seed']}", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
                        cv2.LINE_AA)
            tiles.append(np.vstack([band, im]))
        if tiles:
            cv2.imwrite(str(jdir / f"{cond}.png"), np.hstack(tiles))

    print(f"\n判定圖：{jdir}（每張含同一條件的三個種子）")
    print(f"請於 {csv_path} 中 prompt 欄為 {NEW_NAME} 的列，"
          f"填寫 {'、'.join(k for k, _ in SUBCRITERIA)} 兩欄（1／0）。")
    print(f"**correct 欄留空**，由 --finalize 依連言規則自動回填。")
    print(f"\n判定準則：\n  {CRITERION}")
    print("\n連言規則與其餘三類一致，故所得比率可直接與表 4-8 並列；"
          "\n分項欄則用以判別失敗集中於哪一個服裝概念。")
    print(f"\n填完後執行：python add_fourth_prompt.py --finalize --out {args.out}")


if __name__ == "__main__":
    main()
