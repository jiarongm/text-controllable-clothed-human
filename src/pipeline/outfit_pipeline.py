r"""
outfit_pipeline.py

給 Gradio app 用的「換裝」慢路徑封裝：文字 prompt → 著衣幾何 + UV 貼圖。

這支模組本身在 romp 環境被 import，但實際工作全部 shell out 到別的環境：

    chupa 環境      chupa 生成、UV 展開與多視角渲染、語意標籤
    chupa_tex 環境  Stable Diffusion 貼圖生成（新版 diffusers，跟 chupa 的舊 torch 不相容）

回傳的是「xatlas 重新索引後」的頂點與面，跟 mesh_refine.obj 的索引不同。
呼叫端拿到之後要用這組新頂點重算蒙皮權重，不能沿用舊的。

────────────────────────────────────────────────────────────────
時間預期（這決定 UI 該怎麼設計）
────────────────────────────────────────────────────────────────
  chupa 生成      ~10 分鐘
  render          ~1 分鐘
  semantic        ~10 秒
  generate (SD)   ~10-20 分鐘（50 視角 × 25 步）
  ────────────────────────────────
  合計            ~25-30 分鐘

現場 demo 讓使用者等半小時是不可行的。兩個現實作法：

  1. fast=True —— 砍到 12 個視角、15 步，貼圖生成縮到 3-5 分鐘。
     背面與遮擋區品質會下降，但正面觀感差異不大。

  2. 預生成服裝庫 —— 比照你現有的 shape_assets 做法，先批次跑好一批常見
     服裝，使用者選擇時瞬間切換；只有自訂 prompt 才走完整生成。
     這是 demo 場合唯一實際可行的方式。
"""

import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


# ============================================================
# 環境與路徑（依你的機器調整）
# ============================================================

CHUPA_PROJECT_DIR = "/home/jiaro/projects/chupa"
CHUPA_PYTHON = "/home/jiaro/miniconda3/envs/chupa/bin/python"
CHUPA_TEX_PYTHON = "/home/jiaro/miniconda3/envs/chupa_tex/bin/python"

OUTFIT_ASSETS_DIR = "outfit_assets"     # 預生成服裝庫


def _run(cmd, desc):
    """在指定直譯器下執行，失敗時把 stderr 尾段帶進例外訊息。

    這裡刻意用各環境 python 的絕對路徑，而不是 `conda run`：
    Gradio 是長駐程序，conda run 會多包一層 shell、訊號處理較不乾淨，
    而且絕對路徑本來就是你這支 app 已經在用的模式。
    """
    print(f"[outfit] {desc}")
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=CHUPA_PROJECT_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print("[outfit][stderr]", r.stderr[-3000:])
        raise RuntimeError(f"{desc} 失敗 (returncode={r.returncode})：\n"
                           f"{r.stderr[-800:]}")
    return r.stdout


def _find_new_mesh(subject, since_ts):
    """找出這次生成產出的 mesh_refine.obj。

    只接受修改時間晚於生成開始的檔案 —— 否則同一個 subject 之前的實驗結果
    也會被 glob 到，換裝會安靜地換成上一件衣服。
    """
    pattern = os.path.join(CHUPA_PROJECT_DIR, "results", "renderpeople", "*",
                           subject, "*", "mesh_refine.obj")
    cands = [c for c in glob.glob(pattern) if os.path.getmtime(c) >= since_ts]
    if not cands:
        raise RuntimeError("chupa 生成完成，但找不到新的 mesh_refine.obj。"
                           "路徑規則可能改了，或生成其實沒成功。")
    newest = max(cands, key=os.path.getmtime)
    print(f"[outfit] 新網格: {newest}")
    return newest


# ============================================================
# 主流程
# ============================================================

def build_outfit(prompt, subject, smplx_obj_name, work_dir="tex_work_live",
                 fast=True, seed=42, progress=None, mesh_path=None):
    """跑完整條換裝慢路徑。

    progress:  可選的回呼 (fraction, desc)，接 gradio 的 gr.Progress。
    mesh_path: 給定既有的 mesh_refine.obj 就跳過 chupa 生成，只跑貼圖流程。
               測試貼圖與顯示環節時很有用 —— 省下約 10 分鐘的幾何生成。

    回傳 dict：
        verts    (V,3) float32  世界座標，xatlas 重新索引後
        faces    (F,3) int64
        uv       (V,2) float32
        texture  貼圖 png 的絕對路徑
        mesh_src 原始 mesh_refine.obj 路徑（除錯用）
    """
    def rep(frac, desc):
        print(f"[outfit] {int(frac * 100):3d}%  {desc}")
        if progress:
            progress(frac, desc=desc)

    work = os.path.join(CHUPA_PROJECT_DIR, work_dir)
    smplx_obj = os.path.join(CHUPA_PROJECT_DIR, smplx_obj_name)

    # ── 1. 幾何生成（給了 mesh_path 就跳過）──
    if mesh_path:
        rep(0.40, f"沿用既有幾何：{os.path.basename(mesh_path)}")
    else:
        rep(0.02, "生成著衣幾何（chupa，約 10 分鐘）...")
        t0 = time.time()
        _run([CHUPA_PYTHON, "scripts/chupa.py", "configs/text.yaml",
              f"dataset.subject={subject}",
              f"chupa.prompt={prompt}",
              f"chupa.seed={seed}",
              # 這兩個必須是 0：resample 階段以裸體先驗為條件，
              # 開著會把文字生成的服裝結構洗回長袖長褲，而且不會報錯。
              "chupa.n_resample_from_text=0",
              "chupa.n_resample=0"],
             "chupa 生成")
        mesh_path = _find_new_mesh(subject, t0)

    # ── 2. UV 展開 + 多視角渲染 ──
    rep(0.45, "UV 展開與多視角渲染...")
    cmd = [CHUPA_PYTHON, "uv_texture_bake.py", "--stage", "render",
           "--mesh", mesh_path, "--work", work]
    if fast:
        # 用環境變數覆寫視角數，避免改動腳本本體
        os.environ["UVBAKE_FAST"] = "1"
    _run(cmd, "render")

    # ── 3. 語意標籤 ──
    rep(0.55, "建立衣物語意標籤...")
    _run([CHUPA_PYTHON, "uv_texture_bake.py", "--stage", "semantic",
          "--work", work, "--smplx-obj", smplx_obj,
          "--prompt", prompt],
         "semantic")

    # ── 4. 貼圖生成（唯一需要 chupa_tex 環境的一步）──
    rep(0.60, f"生成貼圖（Stable Diffusion，約 {'3-5' if fast else '10-20'} 分鐘）...")
    tex_out = os.path.join(work, "final")
    _run([CHUPA_TEX_PYTHON, "uv_texture_bake.py", "--stage", "generate",
          "--work", work, "--prompt", prompt, "--out", tex_out],
         "generate")

    # ── 5. 收集結果 ──
    rep(0.95, "載入結果...")
    g = np.load(os.path.join(work, "geometry.npz"))
    texture = os.path.join(work, "texture.png")
    if not os.path.exists(texture):
        raise RuntimeError(f"貼圖生成完成但找不到 {texture}")

    rep(1.0, "完成")
    return {
        "verts": g["verts"].astype(np.float32),
        "faces": g["faces"].astype(np.int64),
        "uv": g["uv"].astype(np.float32),
        "texture": texture,
        "mesh_src": mesh_path,
        "prompt": prompt,
    }


# ============================================================
# 預生成服裝庫
# ============================================================

def save_outfit_asset(outfit, weights, name, assets_dir=OUTFIT_ASSETS_DIR):
    """把一套跑好的服裝存成資產，之後可以瞬間載入。

    現場 demo 不可能讓使用者等半小時，所以實際可行的做法是先批次跑好
    一批常見服裝，使用者選擇時直接載入 —— 跟你現有的 shape_assets 同一個思路。
    """
    d = Path(CHUPA_PROJECT_DIR) / assets_dir / name
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "vertices.npy", outfit["verts"])
    np.save(d / "faces.npy", outfit["faces"])
    np.save(d / "uv.npy", outfit["uv"])
    np.save(d / "weights.npy", weights)

    import shutil
    shutil.copy(outfit["texture"], d / "texture.png")

    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"name": name, "prompt": outfit["prompt"],
                   "n_verts": int(len(outfit["verts"]))}, f,
                  ensure_ascii=False, indent=2)
    print(f"[outfit] 已存成資產: {d}")
    return str(d)


def list_outfit_assets(assets_dir=OUTFIT_ASSETS_DIR):
    """列出所有預生成服裝，回傳 [(顯示名稱, 資料夾路徑), ...]。"""
    out = []
    root = Path(CHUPA_PROJECT_DIR) / assets_dir
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                m = json.load(f)
            out.append((f"{m.get('name')} — {m.get('prompt', '')[:40]}",
                        str(meta_path.parent)))
        except Exception as e:
            print(f"[outfit][WARN] 讀取 {meta_path} 失敗: {e}")
    return out


def load_outfit_asset(asset_dir):
    """載入預生成服裝，回傳 (verts, faces, uv, weights, texture_path)。"""
    d = Path(asset_dir)
    return (np.load(d / "vertices.npy").astype(np.float32),
            np.load(d / "faces.npy").astype(np.int64),
            np.load(d / "uv.npy").astype(np.float32),
            np.load(d / "weights.npy").astype(np.float32),
            str(d / "texture.png"))