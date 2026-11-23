r"""
run_c4_ablation.py —— C4 三臂消融與 CIELAB ΔE 量化

補齊論文 4.5.2 節目前的兩項不足：歸因不明確，以及僅有視覺結果而無量化指標。

一、為何需要第三臂
==================
現行消融只有兩種情形：有語意約束、無語意約束（`--no-semantic`）。問題在於
**後者同時移除了兩件事**：

  1. 區域的**空間界定**——哪裡是上衣、哪裡是下著；
  2. 區域的**顏色資訊**——該區域應該是什麼顏色。

因此兩組的差異無法歸因於其中任一者。論文 4.5.2 節自己也已承認「區域底色是
顏色資訊的主要載體，而非僅是空間上的提示」——既然如此，把改善全歸給
「語意約束」即名實不符。

加入第三臂後構成可分離的比較：

    A  無語意約束（--no-semantic）      無空間、無顏色
    B  中性底色（--neutral-base）        有空間、無顏色
    C  基準（正確底色）                  有空間、有顏色

  A → B 的差異 = **空間約束**的貢獻
  B → C 的差異 = **顏色資訊**的貢獻

二、為何用 CIELAB ΔE
====================
ΔE 是色差的標準度量，其設計目標為「數值差距與人眼所感受的色差成正比」，
較 RGB 的歐氏距離更貼近視覺判斷。本腳本以語意標籤為基準，計算各區域內
生成顏色與該區域目標色的平均 ΔE：

  * ΔE 小 —— 該區域的顏色接近目標，分區顏色正確
  * ΔE 大 —— 顏色偏離目標（可能是無彩度、也可能是被鄰區污染）

**須留意 ΔE 無法區分偏離的方向**：無彩度的灰與被污染的錯誤顏色，都可能
得到相近的 ΔE。故本腳本另報告各區域內的**彩度（chroma）**與**色彩標準差**：

  * 彩度趨近 0 —— 無彩度失效（4.5.2 節於低修補強度下所觀察到者）
  * 區域內標準差大 —— 顏色不均、可能為污染或斑塊

三項指標合看，方能判斷失效的型態，而非僅知「有差異」。

三、實驗控制
============
三臂**均關閉色度校正**（`--recolor 0`）。色度校正必須依賴語意標籤方能運作，
A 組在定義上無法開啟；若僅部分組別開啟，等同同時改變兩個變因。

用法
====
  # 完整執行（生成三臂 + 量測）
  python run_c4_ablation.py --mesh <mesh_refine.obj> \
      --prompt "a navy coat and khaki pants" --out c4_ablation

  # 已生成過，只重新量測
  python run_c4_ablation.py --stage measure --out c4_ablation

環境：本腳本自行指定各階段的直譯器，於任一環境啟動皆可。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
UVBAKE = "uv_texture_bake.py"
PY_CHUPA = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
PY_CHUPA_TEX = os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python")

# 三臂。順序即為「逐步加入資訊」的順序，便於解讀增量。
ARMS = [
    ("A_no_semantic",  ["--no-semantic"],  "無語意約束",   "無空間、無顏色"),
    ("B_neutral_base", ["--neutral-base"], "中性底色",     "有空間、無顏色"),
    ("C_baseline",     [],                 "基準（正確底色）", "有空間、有顏色"),
]

REGION_IDS = {"upper": 1, "lower": 2, "head": 3, "hands": 4, "shoes": 5}
# 僅就服裝區域評估——頭部與手部為膚色，其目標色來自人體而非提示，
# 不屬於本消融所要檢驗的「文字到區域顏色」的對應關係。
EVAL_REGIONS = ["upper", "lower"]


def run(py, stage, work, prompt, extra=None, mesh=None):
    cmd = [py, UVBAKE, "--stage", stage, "--work", str(work), "--prompt", prompt]
    if mesh:
        cmd += ["--mesh", str(mesh)]
    if extra:
        cmd += extra
    t = time.time()
    r = subprocess.run(cmd, cwd=CHUPA_PROJECT_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ {stage} 失敗：{r.stderr[-700:]}")
        return None, time.time() - t
    return r.stdout, time.time() - t


# ============================================================
# 色彩量測
# ============================================================

def rgb_to_lab(img_rgb_u8):
    import cv2
    return cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float64)


def region_stats(tex_rgb, label, atlas_mask, target_rgb):
    r"""回傳某一區域的 ΔE、彩度與色彩標準差。

    ΔE 採 CIE76（LAB 空間的歐氏距離）。選用 CIE76 而非 CIEDE2000，是因為
    本處所比較的是「與目標色的偏離程度」之相對大小，三臂之間採同一定義即可；
    CIEDE2000 的感知均勻性較佳，惟其複雜度對本比較並無額外助益。
    """
    import cv2
    sel = (label == target_rgb["rid"]) & atlas_mask
    n = int(sel.sum())
    if n == 0:
        return None

    lab = rgb_to_lab(tex_rgb)[sel]                      # (n, 3)
    tgt = rgb_to_lab(np.array(target_rgb["rgb"], np.uint8).reshape(1, 1, 3))[0, 0]

    d = lab - tgt[None, :]
    de = np.linalg.norm(d, axis=1)

    # **ΔE 必須拆解，否則無法解讀。**
    # ΔE² = ΔL² + Δa² + Δb²，其中 ΔL 為明度差、後兩項為色度差。
    # 初版僅報 ΔE，結果無法區分「顏色正確了」與「明暗湊巧接近了」——
    # 實測中即出現一組彩度僅 7.5（近乎無彩）卻使 ΔE 大幅下降的情形，
    # 該改善全部來自明度而非顏色。故此處分別報告兩個分量。
    d_l = np.abs(d[:, 0])
    d_ab = np.linalg.norm(d[:, 1:], axis=1)

    chroma = np.linalg.norm(lab[:, 1:] - 128.0, axis=1)  # OpenCV 的 a,b 以 128 為原點
    tgt_chroma = float(np.linalg.norm(tgt[1:] - 128.0))
    return {
        "n_texel": n,
        "delta_e_mean": float(de.mean()),
        "delta_e_p90": float(np.percentile(de, 90)),
        "delta_L_mean": float(d_l.mean()),
        "delta_ab_mean": float(d_ab.mean()),
        "chroma_mean": float(chroma.mean()),
        "target_chroma": tgt_chroma,
        "chroma_ratio": float(chroma.mean() / tgt_chroma) if tgt_chroma > 1e-6 else None,
        "color_std": float(np.std(lab[:, 1:], axis=0).mean()),
    }


def measure_arm(work, target_colors):
    import cv2
    work = Path(work)
    tex_p = work / "texture.png"
    lab_p = work / "label_atlas.npy"
    msk_p = work / "atlas_mask.png"
    for p in (tex_p, lab_p, msk_p):
        if not p.exists():
            print(f"  [略過] 缺少 {p.name}")
            return None

    tex = cv2.cvtColor(cv2.imread(str(tex_p)), cv2.COLOR_BGR2RGB)
    label = np.load(lab_p)
    mask = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE) > 127

    out = {}
    for name in EVAL_REGIONS:
        out[name] = region_stats(
            tex, label, mask,
            {"rid": REGION_IDS[name], "rgb": target_colors[name]})
    return out


# ============================================================
# stages
# ============================================================

def stage_generate_all(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for tag, flags, label, desc in ARMS:
        work = out / tag
        work.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {label}（{desc}）===")

        # render 與 semantic 三臂完全相同，可共用；但為使各臂互相獨立、
        # 避免中間檔互相覆蓋，仍各自執行一次。其成本相對 generate 甚小。
        print("  render ...")
        so, t1 = run(args.py_chupa, "render", work, args.prompt, mesh=args.mesh)
        if so is None:
            continue
        print("  semantic ...")
        extra = ["--smplx-obj", args.smplx_obj] if args.smplx_obj else None
        so, t2 = run(args.py_chupa, "semantic", work, args.prompt, extra=extra)
        if so is None:
            continue

        # 三臂均關閉色度校正——見檔頭「實驗控制」。
        print(f"  generate {' '.join(flags) if flags else '(基準)'} ...")
        so, t3 = run(args.py_chupa_tex, "generate", work, args.prompt,
                     extra=flags + ["--recolor", "0"])
        if so is None:
            continue
        print(f"  完成（{(t1 + t2 + t3) / 60:.1f} 分鐘）")


def stage_measure(args):
    import cv2  # noqa: F401  提前確認可用
    out = Path(args.out)

    # 目標色須與 uv_texture_bake 解析 prompt 所得者一致，否則 ΔE 的基準不對。
    # 此處直接呼叫該模組的解析函式，避免兩處各自維護一份顏色表而失同步。
    sys.path.insert(0, CHUPA_PROJECT_DIR)
    try:
        from uv_texture_bake import parse_prompt_colors
    except ImportError as e:
        sys.exit(f"[錯誤] 無法匯入 uv_texture_bake：{e}")
    target_colors, matched = parse_prompt_colors(args.prompt)
    print(f"目標區域色：" + ", ".join(
        f"{r}={target_colors[r]}" for r in EVAL_REGIONS))
    if not matched:
        print("[警告] 未自提示解析到任何顏色詞——ΔE 的目標色將為預設值，"
              "此時三臂的比較仍有效，但「接近目標色」的解讀須保留。")

    recs = {}
    for tag, _, label, desc in ARMS:
        print(f"\n{label}（{desc}）")
        st = measure_arm(out / tag, target_colors)
        if st is None:
            continue
        recs[tag] = {"label": label, "desc": desc, "regions": st}
        for r in EVAL_REGIONS:
            s = st[r]
            if s is None:
                continue
            cr = f"{100 * s['chroma_ratio']:5.1f}%" if s['chroma_ratio'] else "  —  "
            print(f"  {r:<6} ΔE {s['delta_e_mean']:6.2f}"
                  f"  = 明度 {s['delta_L_mean']:6.2f}"
                  f" + 色度 {s['delta_ab_mean']:6.2f}"
                  f"  ｜彩度 {s['chroma_mean']:5.2f}"
                  f"（目標之 {cr}）"
                  f"  標準差 {s['color_std']:4.2f}")

    # 輸出須自我描述其輸入。缺少這段時，事後無從確認某次結果所用的資產與
    # 提示詞——本研究即曾因此誤述資產的服裝類型，最後係由目標彩度反推色名
    # 方予更正。量測結果與其產生條件應存於同一檔案。
    payload = {
        "_run": {
            "prompt": args.prompt,
            "mesh": str(args.mesh) if args.mesh else "(自動選最新)",
            "smplx_obj": args.smplx_obj or "(uv_texture_bake 預設值)",
            "out": str(out),
            "target_colors": {r: list(target_colors[r]) for r in EVAL_REGIONS},
            "prompt_colors_matched": bool(matched),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "arms": recs,
    }
    (out / "c4_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    emit_report(out, recs, run=payload["_run"])


def emit_report(out, recs, run=None):
    L = []
    L.append("# C4 三臂消融與 ΔE 量化結果\n")
    if run:
        # 報告開頭即列出執行條件，使報告可獨立閱讀而不需回頭查指令
        L.append("\n## 本次執行\n")
        L.append(f"- 提示詞：`{run['prompt']}`")
        L.append(f"- 來源網格：`{run['mesh']}`")
        L.append(f"- SMPL-X：`{run['smplx_obj']}`")
        L.append("- 目標區域色：" + "、".join(
            f"{r}={tuple(c)}" for r, c in run["target_colors"].items()))
        if not run["prompt_colors_matched"]:
            L.append("- **警告：提示詞中未解析到顏色詞，目標色為預設值**")
        L.append(f"- 執行時間：{run['timestamp']}\n")
    L.append("> 三臂均關閉色度校正。ΔE 採 CIE76，以語意標籤界定區域，"
             "僅評估上衣與下著（頭部與手部為膚色，其目標色來自人體而非提示）。\n")

    L.append("\n## 各臂之區域色彩指標\n")
    L.append("> ΔE 已拆為明度分量與色度分量：ΔE² = ΔL² + Δa² + Δb²。"
             "**僅看 ΔE 無法區分「顏色正確」與「明暗湊巧接近」**，故須分列。\n")
    L.append("| 組別 | 資訊 | 區域 | ΔE | 明度差 | 色度差 | 彩度 | 佔目標彩度 |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for tag, _, label, desc in ARMS:
        r = recs.get(tag)
        if not r:
            continue
        for reg in EVAL_REGIONS:
            s = r["regions"].get(reg)
            if not s:
                continue
            cr = f"{100 * s['chroma_ratio']:.0f}%" if s.get("chroma_ratio") else "—"
            L.append(f"| {label} | {desc} | {reg} | {s['delta_e_mean']:.2f} "
                     f"| {s['delta_L_mean']:.2f} | {s['delta_ab_mean']:.2f} "
                     f"| {s['chroma_mean']:.2f} | {cr} |")

    # 增量分解——以**色度差**為主，因其才對應「顏色是否正確」
    a, b, c = (recs.get(t[0]) for t in ARMS)
    if a and b and c:
        L.append("\n## 貢獻的分解（以色度差計）\n")
        L.append("> 此處以**色度差**而非 ΔE 分解。ΔE 含明度分量，"
                 "而明度差異可能來自布料明暗而與顏色定位無關；"
                 "色度差才直接對應「該區域的顏色是否正確」。\n")
        L.append("| 區域 | A→B（空間約束） | B→C（顏色資訊） | A→C（合計） |")
        L.append("|---|---:|---:|---:|")
        for reg in EVAL_REGIONS:
            sa, sb, sc = (x["regions"].get(reg) for x in (a, b, c))
            if not (sa and sb and sc):
                continue
            L.append(f"| {reg} | {sa['delta_ab_mean'] - sb['delta_ab_mean']:+.2f} "
                     f"| {sb['delta_ab_mean'] - sc['delta_ab_mean']:+.2f} "
                     f"| {sa['delta_ab_mean'] - sc['delta_ab_mean']:+.2f} |")
        L.append("\n（正值代表色度差下降，即該區域的顏色更接近目標。）")

        L.append("\n## 對照：以 ΔE 分解（僅供參考，不作為結論依據）\n")
        L.append("| 區域 | A→B | B→C | A→C |")
        L.append("|---|---:|---:|---:|")
        for reg in EVAL_REGIONS:
            sa, sb, sc = (x["regions"].get(reg) for x in (a, b, c))
            if not (sa and sb and sc):
                continue
            L.append(f"| {reg} | {sa['delta_e_mean'] - sb['delta_e_mean']:+.2f} "
                     f"| {sb['delta_e_mean'] - sc['delta_e_mean']:+.2f} "
                     f"| {sa['delta_e_mean'] - sc['delta_e_mean']:+.2f} |")
        L.append("\n**若此表與上表的結論不一致，應以上表為準**，"
                 "並於論文中說明 ΔE 受目標色與中性灰之距離所影響——"
                 "目標色愈接近中性灰者，無彩度輸出的 ΔE 天生愈小，"
                 "該差異反映的是目標色的位置而非方法的優劣。")

        L.append("\n## 判讀指引\n")
        L.append("- **彩度是最直接的指標**：其反映「有沒有產生顏色」，"
                 "不受目標色位置的影響。若 A 與 B 的彩度皆遠低於 C，"
                 "即證實顏色主要來自區域底色而非空間約束。")
        L.append("- **若 B→C 的色度改善遠大於 A→B**：顏色資訊才是主要來源，"
                 "「語意約束」之名不副實，C4 宜改稱「語意區域底色引導」。")
        L.append("- **若 A→B 的色度亦有可觀改善**：空間約束本身確有貢獻"
                 "（例如抑制顏色跨區擴散），則現行名稱可保留，"
                 "惟仍須說明兩者的相對比重。")
        L.append("- **A→B 於某些區域可能為負值**，此非異常。"
                 "目標色愈接近中性灰者（如卡其），無彩度輸出的誤差天生愈小；"
                 "此為目標色位置所致，不應解讀為空間約束有害。")
        L.append("- **區域內標準差**大者代表顏色不均，"
                 "可能為鄰區污染或斑塊狀失效。")

    p = out / "c4_ablation_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[完成] 已寫出 {p}")


def main():
    ap = argparse.ArgumentParser(description="C4 三臂消融與 ΔE 量化")
    ap.add_argument("--stage", choices=["all", "generate", "measure"], default="all")
    ap.add_argument("--mesh", help="來源網格；省略時自動選最新的 mesh_refine.obj")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--smplx-obj")
    ap.add_argument("--out", default="c4_ablation")
    ap.add_argument("--py-chupa", default=PY_CHUPA)
    ap.add_argument("--py-chupa-tex", default=PY_CHUPA_TEX)
    args = ap.parse_args()

    if args.stage in ("all", "generate"):
        if not args.mesh:
            root = Path(CHUPA_PROJECT_DIR)
            cands = sorted(root.glob("results/**/mesh_refine.obj"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not cands:
                sys.exit("[錯誤] 找不到 mesh_refine.obj，請以 --mesh 指定")
            args.mesh = str(cands[0].relative_to(root))
            print(f"[自動選用] --mesh {args.mesh}\n")
        stage_generate_all(args)

    if args.stage in ("all", "measure"):
        stage_measure(args)


if __name__ == "__main__":
    main()