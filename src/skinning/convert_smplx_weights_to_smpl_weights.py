from pathlib import Path
import numpy as np


def main():
    smplx_weight_path = Path("animation_data/chupa_skinning_weights.npy")
    template_path = Path("animation_data/chupa_template_vertices.npy")
    faces_path = Path("animation_data/chupa_faces.npy")

    out_dir = Path("animation_data_smpl")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not smplx_weight_path.exists():
        raise FileNotFoundError(smplx_weight_path)
    if not template_path.exists():
        raise FileNotFoundError(template_path)
    if not faces_path.exists():
        raise FileNotFoundError(faces_path)

    print("[INFO] Loading original SMPL-X transferred weights...")
    w_smplx = np.load(smplx_weight_path).astype(np.float32)
    vertices = np.load(template_path).astype(np.float32)
    faces = np.load(faces_path).astype(np.int64)

    print("[INFO] SMPL-X weights:", w_smplx.shape)
    print("[INFO] vertices:", vertices.shape)
    print("[INFO] faces:", faces.shape)

    if w_smplx.shape[1] != 55:
        raise ValueError(f"Expected SMPL-X weights shape (N, 55), got {w_smplx.shape}")

    # SMPL joint order 24:
    # 0 pelvis
    # 1 left_hip
    # 2 right_hip
    # 3 spine1
    # 4 left_knee
    # 5 right_knee
    # 6 spine2
    # 7 left_ankle
    # 8 right_ankle
    # 9 spine3
    # 10 left_foot
    # 11 right_foot
    # 12 neck
    # 13 left_collar
    # 14 right_collar
    # 15 head
    # 16 left_shoulder
    # 17 right_shoulder
    # 18 left_elbow
    # 19 right_elbow
    # 20 left_wrist
    # 21 right_wrist
    # 22 left_hand
    # 23 right_hand
    #
    # SMPL-X 前 22 個 body joints 與 SMPL 前 22 個大致對應。
    # SMPL-X 22 jaw, 23 left_eye, 24 right_eye → 合併到 SMPL head(15)
    # SMPL-X 25~39 left hand joints → 合併到 SMPL left_hand(22)
    # SMPL-X 40~54 right hand joints → 合併到 SMPL right_hand(23)

    w_smpl = np.zeros((w_smplx.shape[0], 24), dtype=np.float32)

    # body joints: 0 ~ 21
    w_smpl[:, 0:22] = w_smplx[:, 0:22]

    # jaw + eyes -> head
    w_smpl[:, 15] += w_smplx[:, 22]
    w_smpl[:, 15] += w_smplx[:, 23]
    w_smpl[:, 15] += w_smplx[:, 24]

    # left hand detailed joints -> SMPL left_hand
    w_smpl[:, 22] += w_smplx[:, 25:40].sum(axis=1)

    # right hand detailed joints -> SMPL right_hand
    w_smpl[:, 23] += w_smplx[:, 40:55].sum(axis=1)

    # normalize
    w_sum = w_smpl.sum(axis=1, keepdims=True)
    w_smpl = w_smpl / np.maximum(w_sum, 1e-8)

    print("[INFO] Converted SMPL weights:", w_smpl.shape)
    print("[INFO] weight sum min/max:",
          float(w_smpl.sum(axis=1).min()),
          float(w_smpl.sum(axis=1).max()))

    np.save(out_dir / "chupa_smpl_skinning_weights.npy", w_smpl)
    np.save(out_dir / "chupa_template_vertices.npy", vertices)
    np.save(out_dir / "chupa_faces.npy", faces)

    print("[DONE] saved:")
    print(" ", out_dir / "chupa_smpl_skinning_weights.npy")
    print(" ", out_dir / "chupa_template_vertices.npy")
    print(" ", out_dir / "chupa_faces.npy")


if __name__ == "__main__":
    main()
