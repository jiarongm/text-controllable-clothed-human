"""
chupa_reskin.py (v2)
新增：正確處理 body_scale / global_body_translation，
body_pose 切法修正為 bp66[1:22]（丟棄 slot 0，不是丟 slot 21）。
"""

import pickle
import numpy as np
import torch
import smplx
from smplx.lbs import (
    batch_rodrigues,
    vertices2joints,
    blend_shapes,
    batch_rigid_transform,
)

NUM_JOINTS = 55
NUM_BODY_JOINTS = 21


def build_smplx_model(model_dir: str, gender: str = "neutral", num_betas: int = 10):
    model = smplx.create(
        model_dir, model_type="smplx", gender=gender,
        use_pca=True, num_pca_comps=6,
        num_betas=num_betas, batch_size=1,
    )
    model.eval()
    return model


def make_full_pose(global_orient_3, body_pose_63):
    full_pose = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    full_pose[0] = global_orient_3.reshape(3)
    full_pose[1:1 + NUM_BODY_JOINTS] = body_pose_63.reshape(NUM_BODY_JOINTS, 3)
    return full_pose


def compute_joint_transforms(model, betas, full_pose_55x3):
    betas_t = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    full_pose_t = torch.tensor(full_pose_55x3.reshape(1, NUM_JOINTS, 3), dtype=torch.float32)

    v_template = model.v_template.unsqueeze(0)
    shapedirs = model.shapedirs[:, :, : betas_t.shape[-1]]
    J_regressor = model.J_regressor
    parents = model.parents

    v_shaped = v_template + blend_shapes(betas_t, shapedirs)
    J_rest = vertices2joints(J_regressor, v_shaped)

    rot_mats = batch_rodrigues(full_pose_t.view(-1, 3)).view(1, NUM_JOINTS, 3, 3)
    _, A = batch_rigid_transform(rot_mats, J_rest, parents)

    return A.squeeze(0).detach().numpy()


def reskin_vertices(v_bind, weights, G_bind, G_new):
    V = v_bind.shape[0]
    G_bind_inv = np.linalg.inv(G_bind)
    T = np.matmul(G_new, G_bind_inv)
    blended = np.einsum("vj,jab->vab", weights, T)
    v_bind_homo = np.concatenate([v_bind, np.ones((V, 1), dtype=np.float32)], axis=1)
    v_new_homo = np.einsum("vab,vb->va", blended, v_bind_homo)
    return v_new_homo[:, :3]


# ====== 新增：世界座標 <-> 模型原始座標 轉換 ======
def world_to_raw(v_world, scale, translation):
    return (v_world - translation.reshape(1, 3)) / scale


def raw_to_world(v_raw, scale, translation):
    return v_raw * scale + translation.reshape(1, 3)


def load_bind_params(pkl_path):
    """讀取 bind pose 相關的所有參數，回傳統一格式。"""
    with open(pkl_path, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    global_orient = np.array(params["global_orient"], dtype=np.float32).reshape(3)
    betas = np.array(params["betas"], dtype=np.float32).reshape(-1)

    bp66 = np.array(params["body_pose"], dtype=np.float32).reshape(22, 3)
    body_pose_63 = bp66[1:22].reshape(63)   # 修正：丟棄 slot 0

    body_scale = float(np.asarray(params["body_scale"]).reshape(-1)[0])
    translation = np.asarray(params["global_body_translation"], dtype=np.float32).reshape(3)

    full_pose = make_full_pose(global_orient, body_pose_63)
    return {
        "betas": betas,
        "full_pose": full_pose,
        "body_scale": body_scale,
        "translation": translation,
    }


if __name__ == "__main__":
    smplx_model_dir = "src/smpl_related/models"
    bind_pkl_path = "data/renderpeople/smplx/rp_emily_apose_001.pkl"
    chupa_weights_path = "animation_data_emily_apose/chupa_skinning_weights.npy"
    chupa_template_path = "animation_data_emily_apose/chupa_template_vertices.npy"

    bind = load_bind_params(bind_pkl_path)
    model = build_smplx_model(smplx_model_dir)

    G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
    print("G_bind shape:", G_bind.shape)

    weights = np.load(chupa_weights_path).astype(np.float32)
    v_bind_world = np.load(chupa_template_path).astype(np.float32)   # 世界座標(已含 scale/translation)

    # 轉回模型原始座標，才能跟 G_bind（不含 scale/translation）配合做 LBS
    v_bind_raw = world_to_raw(v_bind_world, bind["body_scale"], bind["translation"])

    print("weights shape:", weights.shape)
    print("v_bind_world shape:", v_bind_world.shape)

    # ====== 最小驗證測試：new pose = bind pose ======
    G_new_should_equal_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
    v_test_raw = reskin_vertices(v_bind_raw, weights, G_bind, G_new_should_equal_bind)
    v_test_world = raw_to_world(v_test_raw, bind["body_scale"], bind["translation"])

    diff = np.linalg.norm(v_test_world - v_bind_world, axis=1)
    print("\n=== 最小驗證測試（world 座標） ===")
    print("mean diff:", diff.mean())
    print("max diff :", diff.max())
    print("超過 1mm 的頂點數:", (diff > 0.001).sum(), "/", diff.shape[0])