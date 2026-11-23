"""
make_apose_params_emily.py
以 rp_emily_posed_008 的身形為基礎，產生 A-pose 版本參數。
"""

import pickle
from pathlib import Path
import numpy as np


def main():
    src_pkl_path = Path("data/renderpeople/smplx/rp_emily_posed_008.pkl")
    out_pkl_path = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")

    with open(src_pkl_path, "rb") as f:
        src_params = pickle.load(f, encoding="latin1")

    new_params = {}
    new_params["betas"] = np.asarray(src_params["betas"], dtype=np.float32).reshape(1, 10)
    new_params["global_orient"] = np.zeros((1, 3), dtype=np.float32)
    new_params["body_pose"] = np.zeros((1, 66), dtype=np.float32)
    new_params["left_hand_pose"] = np.zeros((1, 12), dtype=np.float32)
    new_params["right_hand_pose"] = np.zeros((1, 12), dtype=np.float32)
    new_params["jaw_pose"] = np.zeros((1, 3), dtype=np.float32)
    new_params["leye_pose"] = np.zeros((1, 3), dtype=np.float32)
    new_params["reye_pose"] = np.zeros((1, 3), dtype=np.float32)
    new_params["expression"] = np.zeros((1, 10), dtype=np.float32)
    new_params["body_pose_embedding"] = np.zeros((1, 32), dtype=np.float32)
    new_params["body_scale"] = np.asarray(src_params["body_scale"], dtype=np.float32).reshape(-1)
    new_params["global_body_translation"] = np.asarray(
        src_params["global_body_translation"], dtype=np.float32
    ).reshape(3)

    with open(out_pkl_path, "wb") as f:
        pickle.dump(new_params, f)

    print("[DONE] saved:", out_pkl_path)


if __name__ == "__main__":
    main()

