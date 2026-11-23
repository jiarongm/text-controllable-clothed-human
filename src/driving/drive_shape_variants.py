"""
drive_shape_variants.py
讓三個不同身形的著衣網格，用同一段 ROMP 姿態序列驅動。
每個身形使用自己的 bind pkl（betas）、自己的 source mesh、自己的權重。
在 chupa 環境執行。
"""
import glob, os
import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial import cKDTree

from romp_to_chupa_realtime import drive_chupa_with_romp_frame
from chupa_reskin import (
    build_smplx_model, compute_joint_transforms, load_bind_params, world_to_raw,
)

SMPLX_DIR = "src/smpl_related/models"
SHAPES = ["rp_meas_tall_thin_001", "rp_meas_average_001", "rp_meas_short_heavy_001"]
PROMPT_KEY = "a_woman_wearing_a_long_dress"

# ROMP 姿態來源：整段序列（每 N 幀存一次驅動網格）
ROMP_DIR = "/home/jiaro/projects/romp_test/fat_run_output"
SAVE_EVERY = 2
OUT_ROOT = "shape_variants_driven"

model = build_smplx_model(SMPLX_DIR)
lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)


def make_source_mesh(bind):
    """用該身形的 betas + bind pose 生成 SMPL-X 裸體 source（模型座標）→ 轉世界座標"""
    betas_t = torch.tensor(bind["betas"].reshape(1, -1), dtype=torch.float32)
    fp_t = torch.tensor(bind["full_pose"].reshape(1, 55, 3), dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=betas_t,
                    global_orient=fp_t[:, 0],
                    body_pose=fp_t[:, 1:22].reshape(1, 63))
    v = out.vertices[0].numpy()
    v_world = v * bind["body_scale"] + bind["translation"].reshape(1, 3)
    return v_world


def transfer_weights_M1(chupa_v, source_v):
    tree = cKDTree(source_v)
    _, idx = tree.query(chupa_v, k=1)
    w = lbs_w[idx]
    s = w.sum(1, keepdims=True); s[s == 0] = 1
    return (w / s).astype(np.float32)


def find_mesh(subject):
    pat = f"results/renderpeople/*/{subject}/*{PROMPT_KEY}*/mesh_refine.obj"
    cands = glob.glob(pat)
    return max(cands, key=os.path.getmtime) if cands else None


npz_files = sorted(glob.glob(os.path.join(ROMP_DIR, "*.npz")))
print(f"[INFO] ROMP 序列共 {len(npz_files)} 幀")

for subject in SHAPES:
    mesh_path = find_mesh(subject)
    if mesh_path is None:
        print(f"[跳過] {subject}: 找不到著衣網格，請先完成生成")
        continue
    print(f"\n[{subject}] 網格: {mesh_path}")

    bind = load_bind_params(f"data/renderpeople/smplx/{subject}.pkl")
    G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
    source_v = make_source_mesh(bind)

    m = trimesh.load(mesh_path, force="mesh", process=False)
    chupa_v = np.asarray(m.vertices, np.float32)
    faces = np.asarray(m.faces, np.int64)

    weights = transfer_weights_M1(chupa_v, source_v)
    v_bind_raw = world_to_raw(chupa_v, bind["body_scale"], bind["translation"])

    out_dir = f"{OUT_ROOT}/{subject}"
    os.makedirs(out_dir, exist_ok=True)

    for i, npz in enumerate(npz_files):
        if i % SAVE_EVERY != 0:
            continue
        d = np.load(npz, allow_pickle=True)
        r = d["results"].item() if "results" in d.files else d
        if "smpl_thetas" not in r or len(r["smpl_thetas"]) == 0:
            continue
        v_new = drive_chupa_with_romp_frame(
            model, bind, weights, v_bind_raw, G_bind, r["smpl_thetas"][0])
        trimesh.Trimesh(vertices=v_new, faces=faces, process=False).export(
            f"{out_dir}/frame_{i:04d}.obj")
    print(f"  驅動序列已輸出至 {out_dir}/")

print(f"\n[DONE] 三個身形的驅動序列在 {OUT_ROOT}/ 底下各自的資料夾")
