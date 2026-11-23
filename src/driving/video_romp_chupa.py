"""
video_romp_chupa.py (離線版本)
讀取階段1(romp環境)已經產生好的逐幀 .npz 結果，
在 chupa 環境裡逐幀跑蒙皮驅動，不需要 import romp。
"""

import time
import glob
import os
import numpy as np
import trimesh

from romp_to_chupa_realtime import drive_chupa_with_romp_frame
from chupa_reskin import (
    build_smplx_model,
    compute_joint_transforms,
    load_bind_params,
    world_to_raw,
)


def main():
    smplx_model_dir = "src/smpl_related/models"
    bind_pkl_path = "data/renderpeople/smplx/rp_ben_apose_001.pkl"
    chupa_weights_path = "animation_data_apose/chupa_skinning_weights.npy"
    chupa_template_path = "animation_data_apose/chupa_template_vertices.npy"
    chupa_faces_path = "animation_data_apose/chupa_faces.npy"

    bind = load_bind_params(bind_pkl_path)
    model = build_smplx_model(smplx_model_dir)
    G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])

    weights = np.load(chupa_weights_path).astype(np.float32)
    v_bind_world = np.load(chupa_template_path).astype(np.float32)
    faces = np.load(chupa_faces_path).astype(np.int64)
    v_bind_raw = world_to_raw(v_bind_world, bind["body_scale"], bind["translation"])

    romp_output_dir = "/home/jiaro/projects/romp_test/video_output"
    npz_files = sorted(glob.glob(os.path.join(romp_output_dir, "*.npz")))
    npz_files = [f for f in npz_files if "video_results" not in f]   # 排除彙總檔，只用逐幀檔
    print(f"[INFO] 找到 {len(npz_files)} 個逐幀檔案")

    out_dir = "video_driven_output"
    os.makedirs(out_dir, exist_ok=True)

    frame_times = []
    skipped = 0

    for i, npz_path in enumerate(npz_files):
        data = np.load(npz_path, allow_pickle=True)
        results = data["results"].item() if "results" in data.files else data

        if "smpl_thetas" not in results or len(results["smpl_thetas"]) == 0:
            skipped += 1
            continue

        romp_theta_72 = results["smpl_thetas"][0]

        t0 = time.time()
        v_new_world = drive_chupa_with_romp_frame(
            model, bind, weights, v_bind_raw, G_bind, romp_theta_72
        )
        t1 = time.time()
        frame_times.append(t1 - t0)

        if i % 10 == 0:
            trimesh.Trimesh(vertices=v_new_world, faces=faces, process=False).export(
                f"{out_dir}/frame_{i:04d}.obj"
            )

        if i % 30 == 0:
            print(f"[frame {i}/{len(npz_files)}] 蒙皮驅動耗時 {t1-t0:.4f}s ({1.0/(t1-t0):.1f} FPS)")

    frame_times = np.array(frame_times)
    print("\n=== 效能統計(僅蒙皮驅動這一段，不含 ROMP 推論時間) ===")
    print(f"總處理幀數: {len(frame_times)} / {len(npz_files)}(跳過 {skipped} 幀)")
    print(f"平均每幀耗時: {frame_times.mean():.4f}s")
    print(f"平均 FPS: {1.0/frame_times.mean():.2f}")


if __name__ == "__main__":
    main()