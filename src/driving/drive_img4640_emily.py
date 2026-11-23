"""
drive_img4640_emily.py
讀取 IMG_4640 的 ROMP 逐幀結果，用 emily 資產驅動，存出驅動網格供圖 4-6 使用。
在 chupa 環境執行。
"""
import glob, os, time
import numpy as np
import trimesh

from romp_to_chupa_realtime import drive_chupa_with_romp_frame
from chupa_reskin import (
    build_smplx_model, compute_joint_transforms, load_bind_params, world_to_raw,
)

# emily 資產
bind_pkl_path      = "data/renderpeople/smplx/rp_emily_apose_001.pkl"
chupa_weights_path = "animation_data_emily_apose/chupa_skinning_weights.npy"
chupa_template_path= "animation_data_emily_apose/chupa_template_vertices.npy"
chupa_faces_path   = "animation_data_emily_apose/chupa_faces.npy"
smplx_model_dir    = "src/smpl_related/models"

romp_output_dir = "/home/jiaro/projects/romp_test/fat_run_output"
out_dir = "img4640_driven_emily"
os.makedirs(out_dir, exist_ok=True)

bind = load_bind_params(bind_pkl_path)
model = build_smplx_model(smplx_model_dir)
G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
weights = np.load(chupa_weights_path).astype(np.float32)
v_bind_world = np.load(chupa_template_path).astype(np.float32)
faces = np.load(chupa_faces_path).astype(np.int64)
v_bind_raw = world_to_raw(v_bind_world, bind["body_scale"], bind["translation"])

npz_files = sorted(glob.glob(os.path.join(romp_output_dir, "*.npz")))
npz_files = [f for f in npz_files if "video_results" not in f]
print(f"[INFO] 找到 {len(npz_files)} 幀")

SAVE_EVERY = 1   # 每 10 幀存一次，避免太多檔案；要更密可調小
for i, npz_path in enumerate(npz_files):
    data = np.load(npz_path, allow_pickle=True)
    results = data["results"].item() if "results" in data.files else data
    if "smpl_thetas" not in results or len(results["smpl_thetas"]) == 0:
        continue
    romp_theta_72 = results["smpl_thetas"][0]
    v_new = drive_chupa_with_romp_frame(model, bind, weights, v_bind_raw, G_bind, romp_theta_72)
    if i % SAVE_EVERY == 0:
        trimesh.Trimesh(vertices=v_new, faces=faces, process=False).export(
            f"{out_dir}/frame_{i:04d}.obj")
        print(f"  存出 frame_{i:04d}.obj")

print(f"[DONE] 驅動網格存於 {out_dir}/")
