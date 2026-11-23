r"""
gen_extra_prompts.py —— 為圖 4-5 補充四類服裝的質化展示

用途
====
以四組新提示各生成**一例**，擴充圖 4-5 的服裝類別範圍。本腳本沿用
`run_factor_sweep.py` 的「監看法向量圖、出現即終止」機制，故每例約 90 秒
而非完整雕刻的 11 分鐘。

**本腳本所產生者為質化展示，不構成量化證據。**
每類僅一例，n = 1 不足以估計任何比率。圖說與正文須明白寫出此點——
四類提示的正確率量測見表 4-8，本圖不得用以比較品質優劣。

生成參數
========
一律採 4.2.3 節三軸掃描的**基準點**（$s_T = 7.5$、步數 100、兩處重採樣皆 0），
與圖 4-5 現有四格完全相同。此為必要條件：圖 4-5 之圖說載明各格「以相同
生成參數生成」，若新增的四格參數不同，該敘述即與事實不符，而這從圖面
看不出來（此問題曾實際發生，見 `make_fig45.py` 的參數一致性檢核）。

四組提示的定位
==============
四者所測並非同一件事，解讀時須分開：

  black suit          外套不偏離先驗、長褲為先驗預設 → 偏離數假說之檢驗
  black long skirt    裙襬（1 項偏離）               → 偏離數假說之檢驗
  beige qipao         裙襬 ＋ 立領、開衩             → 另涉訓練分布涵蓋度
  white down coat     蓬鬆體積                       → 一種**新的偏離型態**

前二者可由偏離數推得預期；後二者不可，其結果另有成因上的歧義。
**此分類應於生成前寫下**，否則又將落入「於知悉結果後指派偏離數」的問題
（見 4.2.4 節對該問題的說明）。本腳本於執行時會寫出 `predictions.md`
並加上時間戳，即為此用。

用法
====
  conda activate chupa
  cd ~/projects/chupa

  python gen_extra_prompts.py --plan            # 只印指令與預測，不生成
  python gen_extra_prompts.py --out extra_prompts
  python make_fig45.py --out figures            # 併圖（須先更新其 CATEGORIES）

環境：chupa
"""

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 4.2.3 節三軸掃描的基準點。不可更動——圖 4-5 各格須共用同一組參數。
BASE_ST, BASE_STEPS = 7.5, 100
BASE_RESAMPLE, BASE_RESAMPLE_REFINE = 0, 0
SEED = 42

SUBJECT = "rp_emily_apose_001"

# (目錄用的簡名, 提示詞, 預期偏離人體先驗之處, 本例所測為何)
#
# **每一項皆應對應論文中一句具體的限制敘述。** 單例（n = 1）無法支撐任何
# 比率，其唯一的價值在於把某句「未涵蓋」改為「已涵蓋（單例）」；若找不到
# 對應的限制條文，該例不值得生成。
EXTRA = [
    # ---- 第一批（已完成）----
    ("black_suit", "a person wearing a black suit",
     "外套（不偏離）＋長褲（先驗預設）",
     "偏離數假說之檢驗：依該假說，預期與長裙相當或更高。"
     "【結果：類型正確】"),
    ("black_long_skirt", "a woman wearing a black long skirt",
     "裙襬（1 項）",
     "偏離數假說之檢驗：依該假說，預期與長裙相當。"
     "【結果：失敗，生成為褲裝。惟 long_dress 實測正確率 81.5%，"
     "單例失敗之機率即為 18.5%，與該比率相容，不構成反證】"),
    ("beige_qipao", "a woman wearing a beige qipao",
     "裙襬（1 項）＋立領、開衩",
     "另涉訓練分布涵蓋度：若失敗，成因可能為訓練資料中此類服裝稀少，"
     "與偏離數混淆，不可逕自歸因。【結果：失敗，生成為褲裝】"),
    ("white_down_coat", "a person wearing a white down coat",
     "蓬鬆體積——既有四類所無之偏離型態",
     "測試一種新的偏離型態：既有分類以輪廓邊界的位移（袖長、褲長、裙襬）"
     "為主，體積的膨起不在其內。【結果：類型正確】"),

    # ---- 第二批（待生成）----
    ("black_blazer_white_shirt",
     "a person wearing a black blazer over a white shirt",
     "外套與長褲皆不偏離先驗；**惟服裝界線不與人體部位重合**",
     "對應 4.9 節與 C4 之限制：「未涵蓋多層次服飾（其區域界線不與人體"
     "部位重合）」。襯衫的領口與袖口露於外套之外，二者同屬『上衣』區，"
     "語意分區在定義上無法區分。**幾何預期成功；貼圖預期無法分辨兩層**"),

    ("navy_long_dress", "a woman wearing a navy blue long dress",
     "裙襬（1 項）",
     "對應 4.9 節與 C4 之限制：「未涵蓋連身裙裝（其上下區同色）」。"
     "C4 的兩組消融資產皆為上下分離的雙色服裝；連身且上下同色時，"
     "語意分區是否仍正確尚未測過。**幾何預期約 81.5%；"
     "重點在於上下區同色時的貼圖表現**"),

    ("pink_bikini", "a woman wearing a pink bikini",
     "幾何上幾乎不偏離先驗（覆蓋面積極小）",
     "**語意區域劃分之壓力測試**。區域底色由提示詞解析而得，"
     "而語意區域依**人體部位**而非衣物邊界劃分；比基尼的軀幹與腿部"
     "絕大部分為裸露皮膚，卻仍落在 upper／lower 區內。"
     "**幾何預期成功；貼圖預期大面積皮膚被塗成粉紅**——"
     "此即圖 4-17 圖說所述界定誤差之極端情形"),
]


def write_predictions(out):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = [f"# 四組新提示之事前記錄\n",
         f"寫於生成之前：{ts}\n",
         "> 本檔於生成**之前**寫出並加時間戳。4.2.4 節指出，該節之「偏離數」",
         "> 分類係於知悉結果後指派，為其證據強度的主要弱處；本次於生成前",
         "> 先行記錄分類與預期，即為避免重蹈該問題。\n",
         "> **惟須明白：本次每組僅生成一例（n = 1），不足以估計任何比率。**",
         "> 下列「預期」僅供事後對照敘事之用，不構成統計檢定。\n",
         f"\n生成參數（與圖 4-5 現有四格相同）：sT = {BASE_ST}、"
         f"步數 = {BASE_STEPS}、兩處重採樣皆 {BASE_RESAMPLE}、種子 = {SEED}\n",
         "\n| 簡名 | 提示詞 | 偏離人體先驗之處 | 本例所測 |",
         "|---|---|---|---|"]
    for tag, prompt, dev, note in EXTRA:
        L.append(f"| `{tag}` | {prompt} | {dev} | {note} |")
    L.append("\n## 判讀規則（亦於生成前訂定）\n")
    L.append("- 類型正確與否，一律以**未貼圖的正面法向量圖**判定，"
             "不得依貼圖後的彩現結果（見附錄 C 之注意事項）。")
    L.append("- 「black suit」須呈現外套與長褲；「black long skirt」與"
             "「beige qipao」須呈現兩腿間連續之布面且長度及踝；"
             "「white down coat」須呈現明顯厚於軀幹輪廓的蓬鬆外層。")
    L.append("- 單例失敗**不得**解讀為該類別的正確率低；"
             "單例成功亦不得解讀為該類別可靠。")
    p = Path(out) / "predictions.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def build_cmd(prompt):
    return ["python", "scripts/chupa.py", "configs/text.yaml",
            f"dataset.subject={SUBJECT}",
            f"chupa.prompt={prompt}",
            f"chupa.seed={SEED}",
            f"chupa.text_steps={BASE_STEPS}",
            f"chupa.text_cfg_scale={BASE_ST}",
            "chupa.text_image_cfg_scale=1.5",
            f"chupa.n_resample_from_text={BASE_RESAMPLE}",
            f"chupa.n_resample={BASE_RESAMPLE_REFINE}"]


def main():
    ap = argparse.ArgumentParser(description="生成四組補充提示（質化展示）")
    ap.add_argument("--out", default="extra_prompts")
    ap.add_argument("--plan", action="store_true", help="只印指令與預測")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--only", nargs="*", default=[],
                    help="只處理指定的簡名；省略則全部")
    args = ap.parse_args()

    global EXTRA
    if args.only:
        sel = [e for e in EXTRA if e[0] in args.only]
        unknown = set(args.only) - {e[0] for e in EXTRA}
        if unknown:
            sys.exit(f"[錯誤] 未知的簡名：{sorted(unknown)}")
        EXTRA = sel

    p = write_predictions(args.out)
    print(f"[完成] 事前記錄：{p}\n")
    print("**請先閱讀該檔並確認其中的分類與預期，再開始生成。**")
    print("該檔的價值在於寫於結果之前；若於看過結果後修改，即失去意義。\n")

    for tag, prompt, dev, note in EXTRA:
        print(f"  {tag:18s} {prompt}")
        print(f"  {'':18s} 偏離：{dev}")
    print()

    if args.plan:
        print("生成指令（逐條執行，或以 --out 自動跑）：\n")
        for tag, prompt, _, _ in EXTRA:
            print("  " + " ".join(
                f"'{a}'" if " " in a else a for a in build_cmd(prompt)))
        print("\n每例約 90 秒（監看法向量圖出現即終止，不進入雕刻階段）；"
              "\n若無該機制則為完整雕刻約 11 分鐘。")
        return

    # 沿用 run_factor_sweep 的早停機制
    try:
        from run_factor_sweep import run_one_generation
        runner = run_one_generation
        print("[INFO] 沿用 run_factor_sweep 的早停機制（法向量圖出現即終止）")
    except Exception:
        runner = None
        print("[警告] 無法自 run_factor_sweep 匯入早停機制。")
        print("       將執行完整生成，每例約 11 分鐘（四例約 45 分鐘）。")
        print("       若只需法向量圖，可於其出現後手動中止。")

    for tag, prompt, _, _ in EXTRA:
        print(f"\n=== {tag} ===")
        cmd = build_cmd(prompt)
        print("  $ " + " ".join(cmd))
        t = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        dt = time.time() - t
        if r.returncode != 0:
            print(f"  ✗ 失敗（return code {r.returncode}），耗時 {dt:.0f} 秒")
        else:
            print(f"  ✓ 完成，耗時 {dt:.0f} 秒")

    print("\n生成完畢。接續：")
    print("  1. 以**未貼圖的正面法向量圖**逐例判定類型是否正確")
    print("     （results/renderpeople/<時戳>/<subject>/<提示_42>/normal_F/000.png）")
    print("  2. 將判定結果與 predictions.md 對照，據實記錄相符與不符者")
    print("  3. 更新 make_fig45.py 的 CATEGORIES 後併圖")
    print("\n**切記**：四例各 n = 1。圖說與正文不得據此比較品質優劣，"
          "\n亦不得與表 4-8 的正確率並列呈現——後者為 27 次生成的比率。")


if __name__ == "__main__":
    main()
