r"""
make_demo.py

批次產生兩組對照 demo，全部用同一段動作驅動：

  組 A  同一體型 × 三種服裝     → 展示文字對「外觀」的控制力
  組 B  三種體型 × 同一件服裝   → 展示系統對不同身形的適應性

每個變體輸出一個帶動畫與材質的 .glb，並沿 X 軸依序排開，
把同一組的三個檔一起匯入 Blender 就會自動排成一列，直接可以錄影或截圖。

────────────────────────────────────────────────────────────────
執行前必讀
────────────────────────────────────────────────────────────────

* 這支腳本在 chupa 環境跑（需要 smplx / trimesh / scipy），
  貼圖那一步會自己切到 chupa_tex 環境。

* 六個變體約需 1.5 小時，建議用 nohup 或 tmux 掛著跑：

      conda activate chupa
      cd ~/projects/chupa
      nohup python make_demo.py --motion ~/projects/romp_test/img4640_output \
          > demo.log 2>&1 &
      tail -f demo.log

* 強烈建議先跑 --dry-run 確認所有路徑與體型都讀得到（約 1 分鐘，
  不跑 chupa 也不跑 SD，只驗證接線）：

      python make_demo.py --motion ~/projects/romp_test/img4640_output --dry-run

* 組 B 的每個體型都要重跑完整生成。體型改變 betas，網格幾何、UV 展開、
  蒙皮權重全部跟著變，沒辦法共用同一件衣服的資產 —— 兩組成本相同。

* 任何一個變體失敗不會中斷整批，會記錄下來繼續跑下一個，
  最後統一列出成功與失敗清單。跑一個半小時卻因為第五個出錯而全部白費是不可接受的。
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np


# ============================================================
# 設定
# ============================================================

CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
CHUPA_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
CHUPA_TEX_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python")

SMPLX_MODEL_DIR = "src/smpl_related/models"
SHAPE_ASSETS_DIR = "shape_assets"
BASE_SUBJECT = "rp_emily_apose_001"      # 組 A 用的體型，也是建立新 subject 的模板

# 組 A：同體型，三種服裝。刻意選顏色與版型都好分辨的組合。
OUTFITS = [
    ("red_dress",   "a woman wearing a red dress"),
    ("navy_coat",   "a woman wearing a navy coat and khaki trousers"),
    ("green_hood",  "a woman wearing a green hoodie and blue jeans"),
]

# 組 B：三種體型，同一件服裝。
#
# 這裡刻意不用洋裝。組 B 要展示的是「系統對不同身形的適應性」，
# 合身的上衣加長褲最能凸顯輪廓差異；寬鬆的洋裝反而會把身形藏起來，
# 三個體型看起來會很接近。
SHARED_OUTFIT = ("shared_fitted", "a woman wearing a white t-shirt and blue jeans")

X_SPACING = 1.2      # 並排間距（公尺）。人太胖或太高會重疊時調大


# ============================================================
# 工具
# ============================================================

def run(cmd, desc, dry=False):
    print(f"\n  → {desc}")
    print("    $ " + " ".join(str(c) for c in cmd))
    if dry:
        print("    (dry-run，跳過)")
        return
    r = subprocess.run([str(c) for c in cmd], cwd=CHUPA_PROJECT_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} 失敗 (rc={r.returncode})")


def load_shape_assets():
    """讀取預生成體型資產的 betas。

    只取 betas 與身高體重標籤 —— 那些資產的網格是舊流程產生的、不含 UV 與貼圖，
    不能直接沿用，但 betas 可以拿來建立新的 subject 重跑完整流程。
    """
    root = Path(CHUPA_PROJECT_DIR) / SHAPE_ASSETS_DIR
    shapes = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            if "betas" not in m:
                print(f"  [WARN] {meta_path} 沒有 betas，跳過")
                continue
            h = m.get("height_cm") or 0
            w = m.get("weight_kg") or 0
            shapes.append({
                "name": f"h{int(h)}_w{int(w)}",
                "height_cm": h,
                "weight_kg": w,
                "bmi": (w / ((h / 100.0) ** 2)) if h else 0.0,
                "betas": np.asarray(m["betas"], np.float32).reshape(-1),
                # body_scale 與 translation 一定要一起帶。
                # SMPL-X 的 betas 主要控制身材比例，實際身高是靠 body_scale 縮放，
                # 只換 betas 會讓所有體型變成同樣高度，只剩微弱的胖瘦差異。
                "body_scale": m.get("body_scale"),
                "translation": m.get("translation"),
                "dir": str(meta_path.parent),
            })
        except Exception as e:
            print(f"  [WARN] 讀取 {meta_path} 失敗: {e}")
    return shapes


def pick_spread(shapes, k=3, mode="bmi"):
    """挑出用於對照的 k 個體型。

    mode="bmi"     依 BMI 取最低／中位／最高 —— 「瘦／正常／肥胖」的軸線。
    mode="height"  依身高取最矮／中／最高。
    mode="fat"     在身高相近的體型中取 BMI 最低／中／最高。
                   身高被控制住，體型差異不會被身高混淆，對照最乾淨。
    mode="extreme" 取「高瘦 → 矮壯」的對角線。輪廓差異最大，適合 demo：
                   最低 BMI 的取最高者、最高 BMI 的取最矮者，
                   兩個變因同向疊加，視覺對比比單獨變一個明顯得多。
    """
    if len(shapes) <= k:
        return shapes

    if mode == "extreme":
        s = sorted(shapes, key=lambda x: x["bmi"])
        lo_bmi, hi_bmi = s[0]["bmi"], s[-1]["bmi"]
        # 同 BMI 帶內挑身高最極端的，讓兩個變因同向放大差異
        lo = max([x for x in s if x["bmi"] <= lo_bmi + 0.5],
                 key=lambda x: x["height_cm"])
        hi = min([x for x in s if x["bmi"] >= hi_bmi - 0.5],
                 key=lambda x: x["height_cm"])
        mid = sorted(s, key=lambda x: abs(x["bmi"] - (lo_bmi + hi_bmi) / 2))[0]
        return [lo, mid, hi]

    if mode == "height":
        s = sorted(shapes, key=lambda x: x["height_cm"])
    elif mode == "fat":
        # 先找出樣本最密集的身高帶，在該帶內比較胖瘦
        med_h = float(np.median([x["height_cm"] for x in shapes]))
        band = [x for x in shapes if abs(x["height_cm"] - med_h) <= 7]
        s = sorted(band if len(band) >= k else shapes, key=lambda x: x["bmi"])
        if len(band) < k:
            print(f"  [WARN] 身高 {med_h:.0f}±7cm 範圍內只有 {len(band)} 個體型，"
                  f"不足 {k} 個，改用全部樣本依 BMI 排序（身高會混入差異）")
    else:
        s = sorted(shapes, key=lambda x: x["bmi"])

    idxs = [0, len(s) // 2, len(s) - 1][:k]
    return [s[i] for i in idxs]


def fit_shapes(pairs):
    """依指定的身高體重即時擬合 betas，繞過 shape_assets。

    shape_assets 只涵蓋 155–175cm / 46–83kg，對 demo 來說對比不夠強烈。
    這裡直接呼叫既有的擬合工具，可以用更極端的身材（例如 150cm/85kg
    對 190cm/58kg），差異才看得出來。

    body_scale 沿用模板 —— 身高差異由 betas[0] 表現，實測整條 pipeline
    能完整保留（12.6% 的身高差異從 SMPL-X 先驗一路傳到 chupa 輸出）。
    """
    from fit_betas_from_measurements import fit
    out = []
    for h, w in pairs:
        betas, *_ = fit(float(h), float(w))
        betas = np.asarray(betas, np.float32).reshape(-1)
        out.append({
            "name": f"h{int(h)}_w{int(w)}",
            "height_cm": float(h), "weight_kg": float(w),
            "bmi": float(w) / ((float(h) / 100.0) ** 2),
            "betas": betas,
            "body_scale": None, "translation": None,   # 沿用模板
        })
        print(f"  擬合 {h:.0f}cm / {w:.0f}kg  (BMI {out[-1]['bmi']:.1f})  "
              f"|betas|={np.linalg.norm(betas):.3f}")
    return out


def print_shapes(shapes):
    print(f"\n可用體型（共 {len(shapes)} 個）：")
    print(f"  {'名稱':<14}{'身高':>7}{'體重':>7}{'BMI':>7}{'scale':>9}")
    for s in sorted(shapes, key=lambda x: x["bmi"]):
        sc = s.get("body_scale")
        sc_s = f"{float(sc):.3f}" if sc is not None else "—"
        print(f"  {s['name']:<14}{s['height_cm']:>6.0f}cm{s['weight_kg']:>6.0f}kg"
              f"{s['bmi']:>7.1f}{sc_s:>9}")
    print()


def ensure_subject_pkl(name, shape, dry=False, force=False):
    """以 BASE_SUBJECT 為模板，換掉體型參數建立新的 subject 參數檔。

    改 betas、body_scale、translation 三者，保留 body_pose ——
    綁定姿勢一致，體型才是唯一變因。

    body_scale 一定要換：SMPL-X 的 betas 主要控制身材比例，實際身高由
    body_scale 縮放。只換 betas 會讓所有體型長得一樣高，只剩微弱的胖瘦差異。
    """
    tmpl = Path(CHUPA_PROJECT_DIR) / f"data/renderpeople/smplx/{BASE_SUBJECT}.pkl"
    out = Path(CHUPA_PROJECT_DIR) / f"data/renderpeople/smplx/{name}.pkl"
    if out.exists() and not force:
        print(f"  subject 參數檔已存在: {out.name}（要重建請加 --rebuild-subjects）")
        return name
    if dry:
        print(f"  (dry-run) 會建立 subject 參數檔: {out.name}"
              f"  scale={shape.get('body_scale')}")
        return name

    with open(tmpl, "rb") as f:
        data = dict(pickle.load(f, encoding="latin1"))

    orig = np.asarray(data["betas"])
    b = np.zeros(orig.reshape(-1).shape, np.float32)
    betas = shape["betas"]
    n = min(len(betas), len(b))
    b[:n] = betas[:n]
    data["betas"] = b.astype(orig.dtype).reshape(orig.shape)

    if shape.get("body_scale") is not None:
        s = np.asarray(data["body_scale"])
        data["body_scale"] = np.asarray(shape["body_scale"],
                                        dtype=s.dtype).reshape(s.shape)
    if shape.get("translation") is not None:
        t = np.asarray(data["global_body_translation"])
        data["global_body_translation"] = np.asarray(
            shape["translation"], dtype=t.dtype).reshape(t.shape)

    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"  已建立 subject 參數檔: {out.name}  "
          f"scale={shape.get('body_scale')}")
    return name


def ensure_smplx_obj(subject, dry=False):
    """產生該 subject 的 SMPL-X 來源網格（維持原始拓樸）。

    semantic 與 weights 兩個階段都需要一份「與輸出網格對齊、且頂點順序等同
    SMPL-X 模型」的 obj。不同體型的 betas 不同，這份網格也不同，
    所以每個 subject 都要各自產生一份。
    """
    out = Path(CHUPA_PROJECT_DIR) / f"debug_source_smplx_{subject}.obj"
    if out.exists():
        print(f"  SMPL-X 來源網格已存在: {out.name}")
        return out.name
    if dry:
        print(f"  (dry-run) 會產生 SMPL-X 來源網格: {out.name}")
        return out.name

    import torch
    import smplx
    import trimesh

    pkl = Path(CHUPA_PROJECT_DIR) / f"data/renderpeople/smplx/{subject}.pkl"
    with open(pkl, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    betas = np.asarray(p["betas"], np.float32).reshape(1, -1)
    go = np.asarray(p["global_orient"], np.float32).reshape(1, 3)
    bp = np.asarray(p["body_pose"], np.float32).reshape(-1, 3)
    bp63 = bp[1:22].reshape(1, 63)      # 丟棄 slot 0
    scale = float(np.asarray(p["body_scale"]).reshape(-1)[0])
    transl = np.asarray(p["global_body_translation"], np.float32).reshape(1, 3)

    m = smplx.create(os.path.join(CHUPA_PROJECT_DIR, SMPLX_MODEL_DIR),
                     model_type="smplx", gender="neutral", use_pca=True,
                     num_pca_comps=6, num_betas=10, batch_size=1)
    m.eval()
    with torch.no_grad():
        o = m(betas=torch.tensor(betas[:, :10]),
              global_orient=torch.tensor(go),
              body_pose=torch.tensor(bp63))
    v = o.vertices[0].numpy() * scale + transl      # 換到世界座標
    trimesh.Trimesh(vertices=v, faces=m.faces, process=False).export(str(out))
    print(f"  已產生 SMPL-X 來源網格: {out.name} ({len(v)} 頂點)")
    return out.name


# ============================================================
# 單一變體
# ============================================================

def build_variant(tag, subject, prompt, motion, out_dir, x_offset,
                  fps=30, fast=True, dry=False, sd_strength=0.55):
    """跑完一個變體：幾何 → 貼圖 → 權重 → 動畫 glb。"""
    work = f"demo_work/{tag}"
    tex_out = f"{work}/final"
    glb = str(Path(out_dir) / f"{tag}.glb")

    env = dict(os.environ)
    if fast:
        env["UVBAKE_FAST"] = "1"
    os.environ.update(env)

    smplx_obj = ensure_smplx_obj(subject, dry=dry)

    t0 = time.time()
    run([CHUPA_PYTHON, "scripts/chupa.py", "configs/text.yaml",
         f"dataset.subject={subject}", f"chupa.prompt={prompt}",
         "chupa.seed=42",
         # 這兩個必須是 0，否則 resample 會用裸體先驗把文字生成的服裝洗掉
         "chupa.n_resample_from_text=0", "chupa.n_resample=0"],
        "生成幾何 (chupa)", dry)

    # 找出這次產生的網格
    if dry:
        mesh = "<dry-run>"
    else:
        pat = list((Path(CHUPA_PROJECT_DIR) / "results/renderpeople").glob(
            f"*/{subject}/*/mesh_refine.obj"))
        pat = [p for p in pat if p.stat().st_mtime >= t0]
        if not pat:
            raise RuntimeError(f"找不到 {subject} 這次生成的 mesh_refine.obj")
        mesh = str(max(pat, key=lambda p: p.stat().st_mtime))
        print(f"  網格: {mesh}")

    run([CHUPA_PYTHON, "uv_texture_bake.py", "--stage", "render",
         "--mesh", mesh, "--work", work], "UV 展開與多視角渲染", dry)

    run([CHUPA_PYTHON, "uv_texture_bake.py", "--stage", "semantic",
         "--work", work, "--smplx-obj", smplx_obj, "--prompt", prompt],
        "語意標籤", dry)

    run([CHUPA_TEX_PYTHON, "uv_texture_bake.py", "--stage", "generate",
         "--work", work, "--prompt", prompt, "--out", tex_out,
         "--sd-strength", str(sd_strength)],
        "貼圖生成 (Stable Diffusion)", dry)

    run([CHUPA_PYTHON, "drive_avatar.py", "--stage", "weights",
         "--work", work, "--smplx-obj", smplx_obj,
         "--bind-pkl", f"data/renderpeople/smplx/{subject}.pkl"],
        "蒙皮權重轉移", dry)

    run([CHUPA_PYTHON, "drive_avatar.py", "--stage", "export",
         "--work", work,
         "--texture", f"{tex_out}/chupa_textured_texture.png",
         "--bind-pkl", f"data/renderpeople/smplx/{subject}.pkl",
         "--motion", motion, "--fps", str(fps),
         "--x-offset", str(x_offset), "--out", glb],
        "匯出動畫 glb", dry)

    return glb


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True, help="ROMP 輸出資料夾（兩組共用）")
    p.add_argument("--out-dir", default="demo_out")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--full", action="store_true",
                   help="用完整品質（預設快速模式，demo 夠用且快三倍）")
    p.add_argument("--only", choices=["A", "B"], help="只跑其中一組")
    p.add_argument("--list-shapes", action="store_true",
                   help="列出所有可用體型（含 BMI 與 scale）後結束")
    p.add_argument("--shapes", nargs="+",
                   help="指定要用的體型名稱（例如 h155_w46 h165_w60 h175_w83）")
    p.add_argument("--fit-shapes", nargs="+", metavar="H,W",
                   help="直接指定身高,體重即時擬合 betas，繞過 shape_assets。"
                        "例如 --fit-shapes 150,82 170,62 190,58")
    p.add_argument("--shape-mode", default="bmi",
                   choices=["bmi", "height", "fat", "extreme"],
                   help="自動挑選體型的軸線。bmi=瘦/正常/胖；height=矮/中/高；"
                        "fat=在身高相近的體型中比胖瘦（對照最乾淨）；"
                        "extreme=高瘦到矮壯的對角線（輪廓差異最大，適合 demo）")
    p.add_argument("--sd-strength", type=float, default=0.55,
                   help="SD 對區域底色的改動幅度。雙色服裝建議 0.55，"
                        "太高兩色會互相污染")
    p.add_argument("--shared-prompt", default=None,
                   help="覆寫組 B 共用的服裝描述")
    p.add_argument("--rebuild-subjects", action="store_true",
                   help="重建 subject 參數檔（改過體型參數邏輯後必須加）")
    p.add_argument("--dry-run", action="store_true",
                   help="不實際生成，只驗證路徑、體型讀取與指令組裝")
    args = p.parse_args()

    if args.list_shapes:
        print_shapes(load_shape_assets())
        return

    out = Path(CHUPA_PROJECT_DIR) / args.out_dir
    (out / "setA_outfits").mkdir(parents=True, exist_ok=True)
    (out / "setB_shapes").mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Demo 批次生成")
    print(f"  動作      : {args.motion}")
    print(f"  輸出      : {out}")
    print(f"  品質      : {'完整' if args.full else '快速'}")
    if args.dry_run:
        print("  模式      : dry-run（只驗證接線）")
    print("=" * 64)

    jobs = []

    # ── 組 A：同體型 × 三種服裝 ──
    if args.only != "B":
        for i, (tag, prompt) in enumerate(OUTFITS):
            jobs.append({"set": "A", "tag": f"A{i+1}_{tag}",
                         "subject": BASE_SUBJECT, "prompt": prompt,
                         "dir": out / "setA_outfits", "x": i * X_SPACING})

    # ── 組 B：三種體型 × 同一件服裝 ──
    if args.only != "A":
        if args.fit_shapes:
            print("\n即時擬合指定的體型：")
            pairs = [tuple(float(x) for x in s.split(",")) for s in args.fit_shapes]
            shapes = fit_shapes(pairs)
        else:
            shapes = load_shape_assets()
        if not shapes:
            print(f"\n[ERROR] {SHAPE_ASSETS_DIR}/ 底下找不到含 betas 的體型資產。")
            print("        組 B 需要它。可以先用 --only A 跑組 A。")
            if args.only == "B":
                sys.exit(1)
        else:
            if args.fit_shapes:
                picked = shapes                      # 已經是指定好的那幾個
                print(f"\n使用即時擬合的 {len(picked)} 個體型：")
            elif args.shapes:
                by_name = {s["name"]: s for s in shapes}
                missing = [n for n in args.shapes if n not in by_name]
                if missing:
                    print(f"\n[ERROR] 找不到體型: {', '.join(missing)}")
                    print_shapes(shapes)
                    sys.exit(1)
                picked = [by_name[n] for n in args.shapes]
                print(f"\n使用指定的 {len(picked)} 個體型：")
            else:
                picked = pick_spread(shapes, 3, mode=args.shape_mode)
                print(f"\n從 {len(shapes)} 個體型中依 {args.shape_mode} "
                      f"挑出 {len(picked)} 個：")
            for s in picked:
                print(f"  {s['name']}  ({s['height_cm']:.0f}cm / "
                      f"{s['weight_kg']:.0f}kg, BMI {s['bmi']:.1f})")

            tag0, prompt0 = SHARED_OUTFIT
            if args.shared_prompt:
                prompt0 = args.shared_prompt
            for i, s in enumerate(picked):
                subject = f"rp_demo_{s['name']}_001"
                ensure_subject_pkl(subject, s, dry=args.dry_run,
                                   force=args.rebuild_subjects)
                jobs.append({"set": "B", "tag": f"B{i+1}_{s['name']}",
                             "subject": subject, "prompt": prompt0,
                             "dir": out / "setB_shapes", "x": i * X_SPACING})

    print(f"\n共 {len(jobs)} 個變體，"
          f"預估 {len(jobs) * (15 if not args.full else 30)} 分鐘\n")

    ok, failed = [], []
    t_all = time.time()
    for n, j in enumerate(jobs, 1):
        print("\n" + "=" * 64)
        print(f"[{n}/{len(jobs)}] 組 {j['set']} — {j['tag']}")
        print(f"  subject: {j['subject']}")
        print(f"  prompt : {j['prompt']}")
        print("=" * 64)
        t0 = time.time()
        try:
            glb = build_variant(j["tag"], j["subject"], j["prompt"], args.motion,
                                j["dir"], j["x"], fps=args.fps,
                                fast=not args.full, dry=args.dry_run,
                                sd_strength=args.sd_strength)
            ok.append((j["tag"], glb))
            print(f"\n  ✓ 完成 ({(time.time() - t0) / 60:.1f} 分鐘)")
        except Exception as e:
            # 單一變體失敗不中斷整批 —— 跑一個半小時卻因為第五個出錯
            # 而全部白費是不可接受的
            failed.append((j["tag"], str(e)))
            print(f"\n  ✗ 失敗: {e}")
            traceback.print_exc()

    print("\n" + "=" * 64)
    print(f"全部結束，總耗時 {(time.time() - t_all) / 60:.1f} 分鐘")
    print(f"  成功 {len(ok)} / 失敗 {len(failed)}")
    for t, g in ok:
        print(f"    ✓ {t}  →  {g}")
    for t, e in failed:
        print(f"    ✗ {t}  —  {e}")
    print("=" * 64)

    if ok:
        print("\n在 Blender 檢視：")
        print("  File → Import → glTF 2.0，把同一組的檔案全選匯入，")
        print("  它們已經沿 X 軸排開，不會重疊。按空白鍵播放。")
        print(f"\n  組 A（不同服裝）: {out / 'setA_outfits'}")
        print(f"  組 B（不同體型）: {out / 'setB_shapes'}")


if __name__ == "__main__":
    main()