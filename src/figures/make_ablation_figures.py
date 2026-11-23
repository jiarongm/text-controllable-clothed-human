r"""
make_ablation_figures.py —— 產生論文圖 4-10（外觀生成各設計元件的消融對比）

五個設定併為**一張圖**，以基準組為樞紐構成三組正交的單因子比較：

  S1  無語意約束                    (a)
  S2  語意約束、無標籤平滑          (b)      (a)→(b)→(c) 變動語意約束（4.5.2）
  S3  語意約束＋標籤平滑（基準）    (c)      ← 樞紐
  S4  累加完整影像                  (d)      (c)→(d) 變動累加策略（4.5.3）
  S5  基準＋色度校正                (e)      (c)→(e) 變動色度校正（4.5.4）

**為何併圖。** 舊版分為圖 4-10、4-11、4-12 三張，基準組在其中出現三次
（4-10(c)、4-11 右、4-12 左，且為同一次輸出）。併為一張後基準只出現一次，
且三個元件各自相對同一參照點，讀者更易比對。

**基準組於各比較中共用同一次渲染輸出**，此為刻意設計：若各比較分別重跑
基準，擴散模型的隨機性會使同一設定呈現不同外觀，讀者將無從判斷差異來自
被消融的變因或雜訊。

兩項版面上的告誡
================
1. **五格並排於論文頁寬下每格僅約 3 公分。** (d) 與基準的差異為高頻細節
   （皺褶、門襟、縫線），於該尺寸下必然不可見。本腳本因此提供 `--split`
   將五格拆為兩塊排版，以及 `--detail` 另出一張 (c) 對 (d) 的局部放大圖。
   **若不附局部放大圖，(d) 這一格無法支撐 4.5.3 節的論述。**
2. **(b) 與 (c) 的差異本就極小**（見 4.5.2 節：「二者於本圖中的差異甚微」）。
   此為本研究的否定結果之一——標籤平滑的效果主要作用於標籤圖本身，於最終
   貼圖上被稀釋。圖說須明白寫出此點，否則讀者會以為是排版重複。
   欲呈現標籤平滑的實際作用，應改以各設定的 `label_preview.png` 對照。

色度校正之控制
==============
S1 至 S4 均以 `--recolor 0` 關閉色度校正——色度校正須依賴語意標籤方能運作，
S1 在定義上無法開啟；若僅部分組別開啟，等同同時改變兩個變因。S5 為唯一
開啟者，正是 (c)→(e) 所要比較的變因。

用法
====
  # 補跑缺少的設定並產圖（約 8 分鐘）
  python make_ablation_figures.py

  # 只產圖，不重跑任何生成
  python make_ablation_figures.py --render-only

  # 五格拆為 (a)(b)(c) 與 (d)(e) 兩塊，每格較寬
  python make_ablation_figures.py --render-only --split 3

  # 另出 (c) 對 (d) 的局部放大（累加策略的差異為高頻，需放大才看得見）
  python make_ablation_figures.py --render-only --detail

環境：任一（各步驟會自行選用正確的直譯器；渲染需 chupa）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PY_CHUPA = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
PY_CHUPA_TEX = os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python")
BAKE = "uv_texture_bake.py"

# 預設沿用 run_c4_ablation.py 之資產，使本圖與表 4-18 的量化結果同源
DEFAULT_MESH = ("results/renderpeople/20260827-065629/rp_emily_apose_001/"
                "a_person_wearing_a_sleeveless_top_and_long_pants_0042/"
                "mesh_refine.obj")
DEFAULT_PROMPT = "a person wearing a navy sleeveless top and khaki long pants"
DEFAULT_SMPLX = "debug_source_smplx_rp_emily_apose_001.obj"

# (工作目錄, generate 階段的額外旗標, 圖說標籤)
SETTINGS = {
    "S1": ("c4_ablation/A_no_semantic", ["--no-semantic"], "no semantic"),
    "S2": ("abl_fig/S2_no_smooth",      ["--no-smooth"],   "semantic, no smoothing"),
    "S3": ("c4_ablation/C_baseline",    [],                "semantic + smoothing"),
    "S4": ("abl_fig/S4_accum_full",     ["--accumulate-full"], "accumulate full image"),
    "S5": ("abl_fig/S5_recolor",        [],                "with chroma correction"),
}

# S5 為唯一開啟色度校正者
RECOLOR = {"S1": "0", "S2": "0", "S3": "0", "S4": "0", "S5": "0.85"}

# 併圖後的順序與標籤。(c) 為樞紐，故置於中間，使其與 (b) 及 (d) 皆相鄰。
PANELS = [
    ("S1", "(a) no semantic"),
    ("S2", "(b) semantic, no smoothing"),
    ("S3", "(c) semantic + smoothing  [baseline]"),
    ("S4", "(d) accumulate full image"),
    ("S5", "(e) + chroma correction"),
]

# --detail 的比較對：累加策略的差異為高頻，唯有放大方可辨識
DETAIL_PAIR = [("S3", "(c) new pixels only  [baseline]"),
               ("S4", "(d) accumulate full image")]
# 局部放大的取框（相對於 512 畫素的比例）：軀幹中段，含皺褶與下襬
DETAIL_BOX = (0.30, 0.30, 0.72, 0.68)      # x0, y0, x1, y1

CANVAS = 512


def sh(py, args, label):
    print(f"  $ ({Path(py).parent.parent.name}) {' '.join(str(a) for a in args)}")
    r = subprocess.run([py] + [str(a) for a in args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"[中止] {label} 失敗（return code {r.returncode}）")


def ensure(tag, mesh, prompt, smplx, force=False):
    work, flags, _ = SETTINGS[tag]
    w = ROOT / work
    if (w / "texture.png").exists() and not force:
        print(f"[略過] {tag}　{work} 已存在")
        return
    print(f"\n=== {tag}　{work} ===")
    w.mkdir(parents=True, exist_ok=True)
    sh(PY_CHUPA, [BAKE, "--stage", "render", "--mesh", mesh,
                  "--work", work, "--prompt", prompt], f"{tag} render")
    sh(PY_CHUPA, [BAKE, "--stage", "semantic", "--work", work,
                  "--smplx-obj", smplx], f"{tag} semantic")
    sh(PY_CHUPA_TEX, [BAKE, "--stage", "generate", "--work", work,
                      "--prompt", prompt, "--recolor", RECOLOR[tag]] + flags,
       f"{tag} generate")
    sh(PY_CHUPA, [BAKE, "--stage", "export", "--work", work,
                  "--out", work], f"{tag} export")


def load(work):
    import cv2
    w = ROOT / work
    g = np.load(w / "geometry.npz")
    tex = cv2.cvtColor(cv2.imread(str(w / "texture.png")),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return {"verts": g["verts"].astype(np.float64),
            "faces": g["faces"].astype(np.int32),
            "uv": g["uv"].astype(np.float32), "tex": tex}


def block(imgs, labels, label_strip, back_imgs):
    """一塊 = 正面一列（帶標籤）＋ 背面一列。"""
    top = label_strip(imgs, labels)
    bot = label_strip(back_imgs, ["back view"] + [""] * (len(back_imgs) - 1))
    return np.vstack([top, bot])


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-10（五格消融對比）")
    ap.add_argument("--mesh", default=DEFAULT_MESH)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--smplx-obj", default=DEFAULT_SMPLX)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="既有設定亦重跑")
    ap.add_argument("--split", type=int, default=0,
                    help="每塊幾格；預設 0 表示五格一列。設 3 則拆為 3+2 兩塊")
    ap.add_argument("--detail", action="store_true",
                    help="另出 (c) 對 (d) 的局部放大圖")
    args = ap.parse_args()

    if not args.render_only:
        if not (ROOT / args.mesh).exists():
            sys.exit(f"[錯誤] 找不到網格 {args.mesh}")
        for tag in ["S2", "S4", "S5"]:
            ensure(tag, args.mesh, args.prompt, args.smplx_obj, args.force)

    missing = [t for t in SETTINGS
               if not (ROOT / SETTINGS[t][0] / "texture.png").exists()]
    if missing:
        sys.exit(f"[錯誤] 尚缺設定 {missing}；請去掉 --render-only 以補跑")

    import cv2
    import nvdiffrast.torch as dr
    from make_demo_figures import make_view, render, label_strip

    assets = {t: load(SETTINGS[t][0]) for t in SETTINGS}

    # 全部設定共用同一取景。五組的幾何完全相同（同一網格），故取景本應一致；
    # 明確共用可避免任何因浮點差異造成的視覺偏移。
    view = make_view([a["verts"] for a in assets.values()])
    ctx = dr.RasterizeCudaContext()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 先把十張（五設定 × 正背面）渲染出來，後續排版與放大皆重用
    shot = {}
    for tag, _ in PANELS:
        for back in (False, True):
            shot[(tag, back)] = cv2.cvtColor(
                render(assets[tag]["verts"], assets[tag], view, ctx,
                       size=CANVAS, back_view=back), cv2.COLOR_RGB2BGR)

    n = args.split if args.split and args.split > 0 else len(PANELS)
    chunks = [PANELS[i:i + n] for i in range(0, len(PANELS), n)]

    blocks = []
    for ch in chunks:
        front = [shot[(t, False)] for t, _ in ch]
        back = [shot[(t, True)] for t, _ in ch]
        labels = [lb for _, lb in ch]
        blocks.append(block(front, labels, label_strip, back))

    # 各塊寬度可能不同（末塊格數較少），以白底靠左對齊後再堆疊
    w = max(b.shape[1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[1] < w:
            pad = np.full((b.shape[0], w - b.shape[1], 3), 255, np.uint8)
            b = np.hstack([b, pad])
        padded.append(b)
        padded.append(np.full((10, w, 3), 255, np.uint8))     # 塊間留白
    fig = np.vstack(padded[:-1])

    p = out / "fig_4-10_ablation.png"
    cv2.imwrite(str(p), fig)
    print(f"[完成] {p}（{len(PANELS)} 格 × 正背面，"
          f"{len(chunks)} 塊，每塊至多 {n} 格）")

    if n >= 5:
        print("\n  [提醒] 五格並排時每格於論文頁寬下僅約 3 公分。"
              "\n         (d) 與基準的差異為高頻細節，該尺寸下不可見。"
              "\n         建議改用 --split 3，並務必加 --detail 產生局部放大。")

    # ---- 局部放大：(c) 對 (d) ----
    if args.detail:
        x0, y0, x1, y1 = DETAIL_BOX
        cx0, cy0 = int(x0 * CANVAS), int(y0 * CANVAS)
        cx1, cy1 = int(x1 * CANVAS), int(y1 * CANVAS)
        rows = []
        for back in (False, True):
            crops = []
            for tag, _ in DETAIL_PAIR:
                c = shot[(tag, back)][cy0:cy1, cx0:cx1]
                # 放大至兩倍，使高頻細節於印刷尺寸下仍可辨識
                c = cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2),
                               interpolation=cv2.INTER_NEAREST)
                crops.append(c)
            txt = ([lb for _, lb in DETAIL_PAIR] if not back
                   else ["back view", ""])
            rows.append(label_strip(crops, txt))
        dp = out / "fig_4-10_detail_accumulation.png"
        cv2.imwrite(str(dp), np.vstack(rows))
        print(f"[完成] {dp}（取框 {DETAIL_BOX}，放大兩倍，最近鄰插值）")
        print("  放大採最近鄰插值，不作平滑——本圖所比較者為高頻細節，"
              "\n  任何插值平滑都會削弱待觀察的差異。")

    print("\n基準組（S3）於各比較中共用同一次渲染輸出，故三組比較可直接對照。")
    print("圖說須載明：(b) 與 (c) 的差異極小，此為本研究的否定結果"
          "（標籤平滑之效果於最終貼圖上被稀釋），非排版重複。")


if __name__ == "__main__":
    main()
