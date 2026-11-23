r"""
run_ablations.py

跑論文第 4.8 節中三組尚未執行的消融實驗，並直接產出可放入論文的對照圖。

  A. 語意區域約束      無約束 / 有約束但無平滑 / 完整方法      → 圖 4-9
  B. 累加策略          累加完整影像 / 僅累加新生成像素        → 圖 4-10
  C. 色相校正          關閉 / 開啟                            → 新增圖

三組共需七個組別，但其中三個設定完全相同（語意約束開、平滑開、僅累加新像素、
色相校正關）—— 它同時是 A 組的完整方法、B 組的新策略、與 C 組的校正前基準。
故實際只跑五個唯一設定，可省下約兩次生成。

所有設定除了被消融的那一個變因外完全相同（同網格、同提示、同隨機種子、
同視角配置），這是這三組實驗與開發過程中偶然產生的對照最關鍵的差別 ——
早期那些「壞掉的結果」同時混入多個變因，不能當作消融證據。

需特別留意 A 組的設計：整組**全程關閉色相校正**。色相校正需要語意標籤才能
運作，無約束組在定義上無法開啟；若僅部分組別開啟，等同同時改變兩個變因，
該組消融即失去意義。色相校正的效果由 C 組單獨呈現。

在 chupa_tex 環境執行（貼圖生成需要新版 diffusers）。
標籤平滑那一組需要在 chupa 環境重跑 semantic，腳本會自動切換直譯器。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  cd ~/projects/chupa

  # 先確認要用哪個資產。建議用雙色服裝 —— 單色服裝無法暴露顏色錯置問題。
  python run_ablations.py --work demo_work/A2_navy_coat \
      --prompt "a woman wearing a navy coat and khaki trousers" \
      --smplx-obj debug_source_smplx_rp_emily_apose_001.obj \
      --out-dir thesis_figures

輸出：
  thesis_figures/
  ├── _runs/               五個唯一設定的原始輸出
  ├── fig4-9_semantic/     三張貼圖 + 三個 glb + 並排對照圖
  ├── fig4-10_accumulate/  兩張貼圖 + 兩個 glb + 並排對照圖
  ├── fig4-11_recolor/     兩張貼圖 + 兩個 glb + 並排對照圖
  └── ablation_summary.md  實驗設計說明與檔案位置

時間預期：五次生成。以快速模式、既有的 render 輸出計，每次約一小時，
合計約五小時，建議掛在背景跑：

  nohup python run_ablations.py ... > ablation.log 2>&1 &
  tail -f ablation.log

每組除了 UV 貼圖外，另輸出貼回網格的 .glb —— 論文的圖應該用**貼回網格後**的
渲染結果，不是攤平的 UV 貼圖。攤平圖因 xatlas 打包而呈碎片狀，讀者無法判讀。
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2


CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
CHUPA_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
CHUPA_TEX_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python")


def run(cmd, desc, dry=False, fast=True):
    """執行指令。fast=True 時設 UVBAKE_FAST=1，使推論步數與資產產生時一致。

    視角數由 render 階段決定、已固定在資產裡（此處不重跑 render），
    故 UVBAKE_FAST 在這裡只影響 SD 的推論步數（15 對 25）。
    五組消融彼此一致即可，步數高低不影響結論，但會顯著影響總時間。
    """
    print(f"\n  → {desc}")
    print("    $ " + ("UVBAKE_FAST=1 " if fast else "")
          + " ".join(str(c) for c in cmd))
    if dry:
        return
    env = dict(os.environ)
    if fast:
        env["UVBAKE_FAST"] = "1"
    else:
        env.pop("UVBAKE_FAST", None)
    r = subprocess.run([str(c) for c in cmd], cwd=CHUPA_PROJECT_DIR, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 失敗 (rc={r.returncode})")


def snapshot(work, out_dir, name):
    """把這一輪的貼圖與網格另存，避免被下一輪覆蓋。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src_tex = Path(CHUPA_PROJECT_DIR) / work / "texture.png"
    dst_tex = out / f"{name}_texture.png"
    shutil.copy(src_tex, dst_tex)

    # 貼回網格輸出 glb —— 論文的圖要用這個，不是攤平的 UV 貼圖
    src_glb = Path(CHUPA_PROJECT_DIR) / work / "final" / "chupa_textured.glb"
    dst_glb = out / f"{name}.glb"
    if src_glb.exists():
        shutil.copy(src_glb, dst_glb)
        print(f"    已存 {dst_tex.name} 與 {dst_glb.name}")
    else:
        print(f"    已存 {dst_tex.name}")
        print(f"    [WARN] 找不到 {src_glb}，沒有 glb。"
              f"chupa_tex 環境缺 trimesh 時會這樣，")
        print(f"           可執行 pip install trimesh 後重跑 —— "
              f"論文用圖需要 glb，只有貼圖無法判讀。")
    return dst_tex


def make_contact_sheet(paths, labels, out_path, height=512):
    """把數張貼圖並排成一張對照圖，方便快速檢視。

    正式論文用圖建議改用各組的 .glb 在 Blender 以相同視角渲染後並排，
    此處的貼圖並排僅供快速判斷實驗是否成功。
    """
    imgs = []
    for p, lab in zip(paths, labels):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        im = cv2.resize(im, (int(w * height / h), height))
        band = np.full((40, im.shape[1], 3), 255, np.uint8)
        cv2.putText(band, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)
        imgs.append(np.vstack([band, im]))
    if imgs:
        cv2.imwrite(str(out_path), np.hstack(imgs))
        print(f"    對照圖: {out_path}")


# ============================================================
# 三組消融
# ============================================================

# ============================================================
# 五個唯一設定
# ============================================================
#
# 三組消融共需七個組別，但其中三個組別的設定完全相同（語意約束開、平滑開、
# 只累加新像素、色相校正關）—— 它同時是 A 組的完整方法、B 組的新策略、
# 以及 C 組的校正前基準。跑一次共用三處，可省下約兩次生成。
#
# 另一個要點：A 組（語意約束）全部關閉色相校正。色相校正需要語意標籤才能
# 運作，無約束組在定義上就不能開；若 (a) 關而 (b)(c) 開，等於同時動了兩個
# 變因，該組消融即失去意義。色相校正的效果由 C 組單獨呈現。

CONFIGS = {
    # key: (semantic 標籤要不要重建與是否平滑, generate 的額外旗標, 說明)
    "no_semantic":   (None,      ["--no-semantic", "--recolor", "0"],
                      "無語意約束"),
    "no_smooth":     ("nosmooth", ["--recolor", "0"],
                      "有語意約束、無標籤平滑"),
    "baseline":      ("smooth",  ["--recolor", "0"],
                      "語意約束＋平滑＋僅累加新像素（共用基準）"),
    "accum_full":    ("smooth",  ["--recolor", "0", "--accumulate-full"],
                      "累加完整影像"),
    "recolor_on":    ("smooth",  [],
                      "加上色相校正（完整方法）"),
}


def build_config(key, args, out_root):
    """跑一個設定並存檔。回傳存下的貼圖路徑。"""
    sem, flags, desc = CONFIGS[key]
    out = Path(out_root) / "_runs"
    tex_out = f"{args.work}/final"

    # 需要時重建標籤圖（幾秒鐘，在 chupa 環境）
    if sem is not None:
        cmd = [CHUPA_PYTHON, "uv_texture_bake.py", "--stage", "semantic",
               "--work", args.work, "--smplx-obj", args.smplx_obj,
               "--prompt", args.prompt]
        if sem == "nosmooth":
            cmd.append("--no-smooth")
        run(cmd, f"[{key}] 重建標籤圖（{'無' if sem == 'nosmooth' else '含'}平滑）",
            args.dry_run, fast=not args.full)

    gen = [CHUPA_TEX_PYTHON, "uv_texture_bake.py", "--stage", "generate",
           "--work", args.work, "--prompt", args.prompt,
           "--sd-strength", str(args.sd_strength), "--out", tex_out]
    if "--recolor" not in flags:
        gen += ["--recolor", str(args.recolor)]
    gen += flags
    run(gen, f"[{key}] {desc}", args.dry_run, fast=not args.full)

    if args.dry_run:
        return None
    return snapshot(args.work, out, key)


def assemble(out_root, runs, folder, items):
    """把已跑好的設定複製成某一組消融的圖檔並產生對照圖。"""
    out = Path(out_root) / folder
    out.mkdir(parents=True, exist_ok=True)
    paths, labels = [], []
    for key, name, lab in items:
        src = runs.get(key)
        if src is None:
            continue
        dst = out / f"{name}_texture.png"
        shutil.copy(src, dst)
        glb_src = Path(src).with_suffix("").as_posix().replace("_texture", "") + ".glb"
        if Path(glb_src).exists():
            shutil.copy(glb_src, out / f"{name}.glb")
        paths.append(dst)
        labels.append(lab)
    if paths:
        make_contact_sheet(paths, labels, out / "contact_sheet.png")
    return out


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True,
                   help="資產資料夾，例如 demo_work/A2_navy_coat。"
                        "建議用雙色服裝 —— 單色無法暴露顏色錯置")
    p.add_argument("--prompt", required=True)
    p.add_argument("--smplx-obj", default="debug_source_smplx_rp_emily_apose_001.obj")
    p.add_argument("--out-dir", default="thesis_figures")
    p.add_argument("--sd-strength", type=float, default=0.55)
    p.add_argument("--recolor", type=float, default=0.85)
    p.add_argument("--full", action="store_true",
                   help="用 25 步推論而非 15 步。五組彼此一致即可，"
                        "步數不影響消融結論，但會使總時間增為約 1.7 倍")
    p.add_argument("--dry-run", action="store_true",
                   help="只印指令不執行，先確認接線")
    args = p.parse_args()

    out_root = Path(CHUPA_PROJECT_DIR) / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    order = ["no_semantic", "no_smooth", "baseline", "accum_full", "recolor_on"]

    print("=" * 64)
    print("論文消融實驗")
    print(f"  資產   : {args.work}")
    print(f"  提示   : {args.prompt}")
    print(f"  輸出   : {out_root}")
    print(f"  設定數 : {len(order)}（三組消融共用，已去除重複）")
    print(f"  推論步數: {'25（--full）' if args.full else '15（與資產產生時一致）'}")
    print("=" * 64)
    if args.dry_run:
        print("（dry-run，只印指令）")

    t0 = time.time()
    runs, failed = {}, []
    for i, key in enumerate(order, 1):
        print(f"\n{'=' * 64}\n[{i}/{len(order)}] {key}：{CONFIGS[key][2]}\n{'=' * 64}")
        try:
            runs[key] = build_config(key, args, out_root)
        except Exception as e:
            failed.append((key, str(e)))
            print(f"\n  ✗ 失敗: {e}")

    # 依三組消融重新編組
    groups = []
    if not args.dry_run:
        groups.append(("A", "語意區域約束", assemble(
            out_root, runs, "fig4-9_semantic", [
                ("no_semantic", "a_no_semantic", "(a) no semantic constraint"),
                ("no_smooth", "b_no_smoothing", "(b) constraint, no smoothing"),
                ("baseline", "c_full", "(c) constraint + smoothing")])))
        groups.append(("B", "累加策略", assemble(
            out_root, runs, "fig4-10_accumulate", [
                ("accum_full", "a_full_image", "(a) accumulate full image"),
                ("baseline", "b_new_only", "(b) accumulate new pixels only")])))
        groups.append(("C", "色相校正", assemble(
            out_root, runs, "fig4-11_recolor", [
                ("baseline", "a_off", "(a) without hue correction"),
                ("recolor_on", "b_on", "(b) with hue correction")])))

    md = out_root / "ablation_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# 消融實驗輸出\n\n")
        f.write(f"- 資產：`{args.work}`\n- 提示：{args.prompt}\n")
        f.write(f"- SD strength：{args.sd_strength}　色相校正強度：{args.recolor}\n\n")
        f.write("## 實驗設計\n\n")
        f.write("三組消融共需七個組別，但其中三個設定完全相同"
                "（語意約束開、平滑開、僅累加新像素、色相校正關），"
                "故實際只跑五個唯一設定，該共用組別同時作為 A 組的完整方法、"
                "B 組的新策略、與 C 組的校正前基準。\n\n")
        f.write("A 組全程關閉色相校正。色相校正需要語意標籤方能運作，"
                "無約束組在定義上無法開啟；若僅部分組別開啟，"
                "等同同時改變兩個變因，該組消融即失去意義。"
                "色相校正的效果由 C 組單獨呈現。\n\n")
        f.write("## 五個唯一設定\n\n| 設定 | 語意標籤 | 累加策略 | 色相校正 |\n")
        f.write("|---|---|---|---|\n")
        f.write("| no_semantic | 停用 | 僅新像素 | 關 |\n")
        f.write("| no_smooth | 啟用、未平滑 | 僅新像素 | 關 |\n")
        f.write("| baseline | 啟用、已平滑 | 僅新像素 | 關 |\n")
        f.write("| accum_full | 啟用、已平滑 | 完整影像 | 關 |\n")
        f.write("| recolor_on | 啟用、已平滑 | 僅新像素 | 開 |\n\n")
        for key, desc, path in groups:
            f.write(f"## 消融 {key}：{desc}\n\n`{path}`\n\n")
        if failed:
            f.write("## 失敗的設定\n\n")
            for k, e in failed:
                f.write(f"- `{k}`：{e}\n")
        f.write("\n## 論文用圖的產生方式\n\n")
        f.write("各組的 `.glb` 以**相同視角**匯入 Blender 渲染後並排，"
                "不要直接使用攤平的 UV 貼圖 —— 後者因 xatlas 打包而呈碎片狀，"
                "讀者無法判讀。`contact_sheet.png` 僅供快速確認實驗是否成功。\n")

    print(f"\n{'=' * 64}")
    print(f"完成 {len(runs)}/{len(order)} 個設定，"
          f"耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    for k, e in failed:
        print(f"  ✗ {k}: {e}")
    print(f"摘要：{md}")
    print("\n提醒：論文用圖請用各組的 .glb 在 Blender 以相同視角渲染後並排，")
    print("      攤平的 UV 貼圖因打包成碎片狀，不適合直接放進論文。")
    print("=" * 64)


if __name__ == "__main__":
    main()