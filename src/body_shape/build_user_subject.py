r"""
build_user_subject.py —— 由身高體重建立可供 Chupa 生成的 subject

用途：為圖 4-7 之受測者（或任何使用者）建立對應體型的 SMPL-X 先驗，
供後續生成著衣網格、完成「真人 → 個人化資產 → 驅動 → 合成」的整合展示。

一支腳本串起三段既有流程，省去手動改檔與路徑對位：
  1. fit_betas_from_measurements.fit()        身高體重 → betas
  2. 以 rp_emily_apose_001.pkl 為模板寫出 subject pkl
  3. 印出 Chupa 的生成指令（含 3.2.2 節所確立的參數）

**先跑這支再決定要不要生成。** 完整生成需十餘分鐘，而本腳本於三十秒內
即可回報擬合誤差與 |β| 的量級——若該體型落在已驗證範圍之外，生成品質
可能下降（4.2.5 節），屆時可先調整預期或改選體型。

已驗證的範圍
============
表 4-14 之 11 組體型，其形狀參數分量實測介於 −1.79 至 +2.06。
4.2.5 節另載：偏離平均體型較遠者，服裝結構的生成品質有下降的傾向，
故體型庫採較溫和的參數範圍。本腳本據此於 |β| 超出該範圍時提出警告——
**警告不代表不能做，而是代表若結果不佳，成因已知且應據實報告**。

用法
====
  conda activate chupa
  cd ~/projects/chupa
  python build_user_subject.py --height 155 --weight 80 --name user_155_80

環境：chupa
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TEMPLATE_PKL = ROOT / "data/renderpeople/smplx/rp_emily_apose_001.pkl"
BETAS_DIR = ROOT / "measurement_fit"

# 表 4-14 所載之已驗證範圍
VERIFIED_LO, VERIFIED_HI = -1.791, 2.060

# 3.2.2 節所確立的生成參數（停用兩處重採樣）
GEN_CMD = """python scripts/chupa.py configs/text.yaml \\
    dataset.subject={subject} \\
    'chupa.prompt={prompt}' \\
    chupa.seed=42 \\
    chupa.text_steps=100 chupa.text_cfg_scale=7.5 chupa.text_image_cfg_scale=1.5 \\
    chupa.n_resample_from_text=0 chupa.n_resample=0"""

DEFAULT_PROMPT = "a woman wearing a white t-shirt and blue jeans"


def main():
    ap = argparse.ArgumentParser(description="由身高體重建立 subject")
    ap.add_argument("--height", type=float, required=True, help="公分")
    ap.add_argument("--weight", type=float, required=True, help="公斤")
    ap.add_argument("--name", required=True,
                    help="subject 名稱（不含 rp_ 前綴與 _001 後綴）")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--fit-only", action="store_true",
                    help="只擬合並回報，不寫出 subject pkl")
    args = ap.parse_args()

    if not TEMPLATE_PKL.exists():
        sys.exit(f"[錯誤] 找不到模板 {TEMPLATE_PKL}")

    try:
        from fit_betas_from_measurements import fit
    except ImportError as e:
        sys.exit(f"[錯誤] 無法匯入 fit_betas_from_measurements：{e}")

    bmi = args.weight / (args.height / 100.0) ** 2
    print(f"目標：身高 {args.height:.0f} cm、體重 {args.weight:.0f} kg"
          f"（BMI {bmi:.1f}）\n")

    betas, verts, faces = fit(args.height, args.weight, iters=args.iters)
    betas = np.asarray(betas).reshape(-1)[:10]

    # ---- 與已驗證範圍比對 ----
    lo, hi = float(betas.min()), float(betas.max())
    mx = float(np.abs(betas).max())
    print(f"\n形狀參數：最小 {lo:+.3f}、最大 {hi:+.3f}、絕對值上限 {mx:.3f}")
    print(f"已驗證範圍（表 4-14，11 組體型）：{VERIFIED_LO:+.3f} 至 {VERIFIED_HI:+.3f}")

    out_of_range = lo < VERIFIED_LO or hi > VERIFIED_HI
    if out_of_range:
        print("\n  ⚠ **本體型落在已驗證範圍之外。**")
        print("    4.2.5 節載明：偏離平均體型較遠者，服裝結構的生成品質有下降"
              "的傾向。\n    本次生成仍可進行，惟若結果不佳，**成因已知**——"
              "應據實報告為\n    「生成品質隨體型偏離程度下降」的實例，"
              "而非更換體型直到好看為止。")
    else:
        print("\n  ✓ 落在已驗證範圍之內，生成品質應與表 4-13 各組相當。")

    BETAS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(BETAS_DIR / f"{args.name}_betas.npy", betas)
    try:
        import trimesh
        trimesh.Trimesh(verts, faces, process=False).export(
            str(BETAS_DIR / f"{args.name}.obj"))
        print(f"\n[完成] betas 與裸體網格：{BETAS_DIR}/{args.name}_*")
    except Exception as e:
        print(f"\n[完成] betas：{BETAS_DIR}/{args.name}_betas.npy"
              f"（網格匯出略過：{e}）")

    if args.fit_only:
        print("\n（--fit-only，未寫出 subject pkl）")
        return

    # ---- 寫出 subject pkl ----
    subject = f"rp_{args.name}_001"
    out_pkl = TEMPLATE_PKL.parent / f"{subject}.pkl"
    if out_pkl.exists():
        print(f"\n[停止] {out_pkl} 已存在。"
              "\n       覆寫既有 subject 會使引用該 subject 的既有資產"
              "與其先驗不一致；\n       請改用其他 --name，或先行確認可覆寫後手動刪除。")
        return

    with open(TEMPLATE_PKL, "rb") as f:
        template = pickle.load(f, encoding="latin1")
    orig = np.asarray(template["betas"])
    data = dict(template)
    data["betas"] = betas.astype(orig.dtype).reshape(orig.shape)
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)
    print(f"[完成] subject：{out_pkl}")
    print(f"       （沿用模板之 A-pose 姿態、body_scale 與 translation，"
          f"僅替換 betas）")

    # ---- 後續指令 ----
    print("\n" + "=" * 68)
    print("接續步驟")
    print("=" * 68)
    print("\n一、生成著衣網格（約 11 分鐘；參數為 3.2.2 節所確立者）：\n")
    print("  " + GEN_CMD.format(subject=subject, prompt=args.prompt))
    print("\n二、建成可驅動資產（品質組態，約 3 分鐘）：\n")
    print("  unset UVBAKE_FAST")
    print(f"  python build_asset.py --match {args.prompt.split()[-1]} \\")
    print(f"      --work demo_work/U_{args.name}")
    print("\n  **--match 須確認只命中本次生成**（同一提示詞可能已有其他"
          "體型的執行）；\n  可先以 python build_asset.py --list 檢視，"
          "必要時改用時間戳比對。")
    print("\n三、以該受測者的動作驅動並合成：\n")
    print("  python prep_romp_motion.py --video motion_largebody/F3.mp4 \\")
    print("      --name largebody --out motion_out")
    print(f"  python render_overlay.py --video motion_largebody/F3.mp4 \\")
    print(f"      --motion motion_out/largebody \\")
    print(f"      --work demo_work/U_{args.name} \\")
    print(f"      --bind-pkl data/renderpeople/smplx/{subject}.pkl \\")
    print(f"      --out /tmp/overlay_{args.name}.mp4")
    print("\n**第三步的 --bind-pkl 必須是本 subject 的 pkl**，"
          "不可沿用 emily——\n綁定姿勢的體型須與資產的生成體型一致，"
          "否則蒙皮權重轉移會對位到錯誤的身形。")


if __name__ == "__main__":
    main()
