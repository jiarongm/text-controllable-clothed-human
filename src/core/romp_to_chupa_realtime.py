import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from chupa_reskin import (
    build_smplx_model,
    compute_joint_transforms,
    reskin_vertices,
    load_bind_params,
    world_to_raw,
    raw_to_world,
    make_full_pose,
)

# ROMP 相機空間 -> 我們驗證過的 SMPL-X 世界座標系，校正矩陣
R_CW = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def romp_smpl_to_smplx_pose(romp_theta_72: np.ndarray):
    """
    romp_theta_72: (72,) = global_orient(3) + body_pose(69, 23 joints)
    加入座標系校正：ROMP 是相機空間(Y朝下)，SMPL-X世界座標是Y朝上，
    差一個繞X軸180度的旋轉，只需要校正 global_orient（根旋轉），
    body_pose 是相對父關節的局部旋轉，不受世界座標系影響，不用校正。
    """
    romp_theta_72 = romp_theta_72.reshape(24, 3)

    romp_global_orient_mat = R_scipy.from_rotvec(romp_theta_72[0]).as_matrix()
    corrected_global_orient_mat = R_CW @ romp_global_orient_mat
    global_orient = R_scipy.from_matrix(corrected_global_orient_mat).as_rotvec().astype(np.float32)

    body_pose_63 = romp_theta_72[1:22].reshape(63)   # joint 1~21，捨棄22,23（同前）

    return global_orient, body_pose_63


def drive_chupa_with_romp_frame(model, bind, weights, v_bind_raw,
                                 G_bind, romp_theta_72, betas_override=None):
    global_orient, body_pose_63 = romp_smpl_to_smplx_pose(romp_theta_72)
    betas = betas_override if betas_override is not None else bind["betas"]

    new_full_pose = make_full_pose(global_orient, body_pose_63)
    G_new = compute_joint_transforms(model, betas, new_full_pose)

    v_new_raw = reskin_vertices(v_bind_raw, weights, G_bind, G_new)
    v_new_world = raw_to_world(v_new_raw, bind["body_scale"], bind["translation"])
    return v_new_world


if __name__ == "__main__":
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

    romp_result = np.load("/home/jiaro/projects/romp_test/run.npz", allow_pickle=True)["results"].item()
    romp_theta_72 = romp_result["smpl_thetas"][0]

    v_new_world = drive_chupa_with_romp_frame(
        model, bind, weights, v_bind_raw, G_bind, romp_theta_72
    )

    import trimesh
    trimesh.Trimesh(vertices=v_new_world, faces=faces, process=False).export(
        "debug_romp_driven_chupa_v2.obj"
    )
    print("[DONE] saved debug_romp_driven_chupa_v2.obj")