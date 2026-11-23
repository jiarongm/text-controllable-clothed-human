r"""
seed_outfit.py

在 romp 環境執行。用途有兩個，都是為了讓 Gradio app 能被實際測試：

  1. 把「現在正在用的」著衣網格烘出貼圖並存成服裝資產。
     Gradio 的服裝庫一開始是空的，快速換裝頁籤沒東西可選；
     而完整生成要等 25 分鐘才知道 UI 對不對。這支腳本沿用既有幾何、
     只跑貼圖流程，把等待縮到幾分鐘。

  2. 解決 UV_FLIP_V 的不確定性。
     OBJ 的 v 軸原點在下、glTF 在上，而 trimesh 匯出 glTF 時是否自動翻轉
     並不明確。這支腳本會同時匯出翻轉與不翻轉兩個 glb，你用 Blender
     各開一次，看哪個正確，再把 gradio 檔裡的 UV_FLIP_V 設成對應的值。

     判斷方式：正確的那個是乾淨的區域分色（上衣一色、褲子一色、頭手膚色）；
     錯誤的那個會是顏色種類正確但位置全部錯亂的拼貼亂碼。

用法：
  conda activate romp
  cd ~/projects/chupa

  # 從既有的 animation_data 資產烘貼圖（不重跑 chupa）
  python seed_outfit.py --from-npy animation_data_emily_apose \
      --prompt "a woman wearing a red dress" --name red_dress

  # 或指定一個既有的 mesh_refine.obj
  python seed_outfit.py --mesh results/.../mesh_refine.obj \
      --prompt "a person wearing a navy coat and khaki trousers" --name navy_coat

  # 只想確認 UV 翻轉方向，不存資產
  python seed_outfit.py --from-npy animation_data_emily_apose \
      --prompt "..." --check-uv-only
"""

import argparse
import os

import numpy as np
import trimesh
from PIL import Image

import outfit_pipeline as op


CHUPA_PROJECT_DIR = op.CHUPA_PROJECT_DIR
CHUPA_SUBJECT = "rp_emily_apose_001"
SOURCE_SMPLX_OBJ_NAME = "debug_source_smplx_rp_emily_apose_001.obj"
SMPLX_MODEL_DIR = "src/smpl_related/models"


def npy_to_obj(npy_dir, out_path):
    """把 animation_data 裡的 npy 頂點與面還原成 obj。

    貼圖流程的入口需要一個網格檔，而既有資產是拆開存的 npy。
    這樣就不必為了測試重跑一次 chupa。
    """
    d = os.path.join(CHUPA_PROJECT_DIR, npy_dir)
    v = np.load(os.path.join(d, "chupa_template_vertices.npy")).astype(np.float64)
    f = np.load(os.path.join(d, "chupa_faces.npy")).astype(np.int64)
    trimesh.Trimesh(vertices=v, faces=f, process=False).export(out_path)
    print(f"[seed] 已從 {npy_dir} 還原網格 → {out_path} "
          f"({len(v)} 頂點, {len(f)} 面)")
    return out_path


def compute_weights(target_verts, smplx_model_dir, source_obj):
    """M1 最近頂點權重轉移（與 Gradio app 中的實作一致）。"""
    import smplx
    from scipy.spatial import cKDTree

    model = smplx.create(os.path.join(CHUPA_PROJECT_DIR, smplx_model_dir),
                         model_type="smplx", gender="neutral",
                         use_pca=True, num_pca_comps=6,
                         num_betas=10, batch_size=1)
    model.eval()
    lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)

    src = trimesh.load(os.path.join(CHUPA_PROJECT_DIR, source_obj),
                       force="mesh", process=False)
    src_v = np.asarray(src.vertices, np.float32)
    if len(src_v) != len(lbs_w):
        raise ValueError(f"source obj 有 {len(src_v)} 頂點，"
                         f"SMPL-X 模型是 {len(lbs_w)}。obj 必須維持原始拓樸。")

    dist, idx = cKDTree(src_v).query(target_verts, k=1)
    print(f"[seed] 最近點距離 mean={dist.mean():.4f} max={dist.max():.4f}")

    w = lbs_w[idx]
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = (w / s).astype(np.float32)
    if np.isnan(w).any():
        raise ValueError("權重含 NaN")
    return w


def export_uv_test_pair(verts, faces, uv, texture_path, out_dir="/tmp"):
    """匯出翻轉與不翻轉兩個 glb，讓使用者用眼睛決定 UV_FLIP_V。

    這個參數我無法從程式碼推定 —— trimesh 各版本對 glTF 的 UV 軸處理不同，
    而錯了不會報錯，只會產生看起來像拼貼亂碼的模型。直接兩個都做出來比較快。
    """
    from trimesh.visual import TextureVisuals

    img = Image.open(texture_path).convert("RGB")
    paths = {}
    for flip in (False, True):
        u = uv.copy()
        if flip:
            u[:, 1] = 1.0 - u[:, 1]
        m = trimesh.Trimesh(vertices=verts, faces=faces,
                            visual=TextureVisuals(uv=u, image=img),
                            process=False)
        p = os.path.join(out_dir, f"uvtest_flip_{str(flip).lower()}.glb")
        m.export(p)
        paths[flip] = p

    print("\n" + "=" * 60)
    print("UV 方向測試檔已匯出，用 Blender 各開一次：")
    print(f"  UV_FLIP_V = False  →  {paths[False]}")
    print(f"  UV_FLIP_V = True   →  {paths[True]}")
    print()
    print("正確的那個：乾淨的區域分色（上衣一色、褲子一色、頭手膚色）。")
    print("錯誤的那個：顏色種類正確但位置全部錯亂的拼貼亂碼。")
    print("把 gradio_realtime_romp_chupa.py 裡的 UV_FLIP_V 設成正確的那個值。")
    print("=" * 60 + "\n")
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, help="服裝描述（決定貼圖顏色）")
    p.add_argument("--name", default=None, help="存入服裝庫的名稱")
    p.add_argument("--from-npy", default=None,
                   help="從 animation_data 資料夾還原網格（例如 animation_data_emily_apose）")
    p.add_argument("--mesh", default=None, help="直接指定 mesh_refine.obj")
    p.add_argument("--subject", default=CHUPA_SUBJECT)
    p.add_argument("--smplx-obj", default=SOURCE_SMPLX_OBJ_NAME)
    p.add_argument("--work", default="tex_work_seed")
    p.add_argument("--full", action="store_true",
                   help="用完整品質（預設快速模式，測試階段夠用）")
    p.add_argument("--check-uv-only", action="store_true",
                   help="只做 UV 方向測試，不存服裝資產")
    args = p.parse_args()

    if not args.mesh and not args.from_npy:
        p.error("要給 --mesh 或 --from-npy 其中一個")

    mesh_path = args.mesh
    if args.from_npy:
        mesh_path = os.path.join(CHUPA_PROJECT_DIR, "_seed_mesh.obj")
        npy_to_obj(args.from_npy, mesh_path)

    print(f"[seed] 開始烘貼圖（{'完整' if args.full else '快速'}模式）...")
    outfit = op.build_outfit(
        prompt=args.prompt,
        subject=args.subject,
        smplx_obj_name=args.smplx_obj,
        work_dir=args.work,
        fast=not args.full,
        mesh_path=mesh_path,      # 跳過 chupa
    )

    export_uv_test_pair(outfit["verts"], outfit["faces"],
                        outfit["uv"], outfit["texture"])

    if args.check_uv_only:
        print("[seed] --check-uv-only：不存資產，結束。")
        return

    if not args.name:
        print("[seed] 沒給 --name，不存入服裝庫。")
        return

    print("[seed] 計算蒙皮權重...")
    w = compute_weights(outfit["verts"], SMPLX_MODEL_DIR, args.smplx_obj)
    path = op.save_outfit_asset(outfit, w, args.name)
    print(f"[seed] 完成。啟動 Gradio 後在「快速換裝」頁籤按重新整理就會看到「{args.name}」。")


if __name__ == "__main__":
    main()
