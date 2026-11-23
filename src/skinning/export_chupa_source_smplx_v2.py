from pathlib import Path
import pickle
import numpy as np
import torch
import smplx
import trimesh


def to_tensor(data, key, default_shape):
    if key not in data:
        return torch.zeros(default_shape, dtype=torch.float32)
    arr = np.asarray(data[key]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    return torch.tensor(arr, dtype=torch.float32)


def main():
    smplx_model_dir = Path("src/smpl_related/models")
    smplx_param_path = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
    out_path = Path("debug_source_smplx_rp_emily_apose_001.obj")

    with open(smplx_param_path, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    global_orient = to_tensor(params, "global_orient", (1, 3))
    betas = to_tensor(params, "betas", (1, 10))

    # ====== 關鍵修正 1：body_pose 正確切法，比照 utils.py ======
    body_pose_66 = to_tensor(params, "body_pose", (1, 66))
    if body_pose_66.shape[1] == 66:
        body_pose = body_pose_66[:, 3:]   # 丟棄前3個值(slot 0)，保留63個(slot 1~21)
    else:
        body_pose = body_pose_66

    # ====== 關鍵修正 2：手部姿勢用 PCA，維度比照 utils.py 的 [:, :6] ======
    left_hand_pose_raw = to_tensor(params, "left_hand_pose", (1, 12))
    right_hand_pose_raw = to_tensor(params, "right_hand_pose", (1, 12))
    left_hand_pose = left_hand_pose_raw[:, :6]
    right_hand_pose = right_hand_pose_raw[:, :6]

    jaw_pose = to_tensor(params, "jaw_pose", (1, 3))
    leye_pose = to_tensor(params, "leye_pose", (1, 3))
    reye_pose = to_tensor(params, "reye_pose", (1, 3))
    expression = to_tensor(params, "expression", (1, 10))

    # ====== 關鍵修正 3：讀取 body_scale / global_body_translation ======
    body_scale = float(np.asarray(params["body_scale"]).reshape(-1)[0])
    global_body_translation = torch.tensor(
        np.asarray(params["global_body_translation"], dtype=np.float32).reshape(1, 3)
    )

    print("[INFO] body_pose(修正後):", body_pose.shape)
    print("[INFO] body_scale:", body_scale)
    print("[INFO] global_body_translation:", global_body_translation.numpy())

    # ====== 關鍵修正 4：model 要用 use_pca=True, num_pca_comps=6，跟 utils.py 一致 ======
    model = smplx.create(
        model_path=str(smplx_model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=True,
        num_pca_comps=6,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=1,
    )

    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
        )

    verts_raw = output.vertices[0].detach().cpu().numpy()

    # ====== 關鍵修正 5：套用 scale + translation（yaw 不用做，pipeline 內部會自己抵銷）======
    verts_final = verts_raw * body_scale + global_body_translation.numpy()

    faces = model.faces
    mesh = trimesh.Trimesh(vertices=verts_final, faces=faces, process=False)
    mesh.export(out_path)

    print("[DONE] saved:", out_path)
    print("[INFO] size (max-min):", verts_final.max(0) - verts_final.min(0))
    print("[INFO] centroid:", verts_final.mean(0))


if __name__ == "__main__":
    main()
