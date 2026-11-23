"""
build_outfit_change_figure.py
為圖 4-7 準備素材：對指定的多件服裝，各自做 emily 權重轉移，
並驅動到同一個姿勢，匯出網格供並排截圖。
在 chupa 環境執行。
"""
import glob, os
import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial import cKDTree
from smplx.lbs import batch_rodrigues, vertices2joints, blend_shapes, batch_rigid_transform

from romp_to_chupa_realtime import drive_chupa_with_romp_frame
from chupa_reskin import (
    build_smplx_model, compute_joint_transforms, load_bind_params, world_to_raw,
)

SUBJECT = "rp_emily_apose_001"
SOURCE_OBJ = "debug_source_smplx_rp_emily_apose_001.obj"
BIND_PKL = "data/renderpeople/smplx/rp_emily_apose_001.pkl"
SMPLX_DIR = "src/smpl_related/models"

# 要對比的服裝：標籤 -> prompt 資料夾關鍵字（用來從 results 找對應 mesh）
OUTFITS = {
    "long_dress":   "a_woman_wearing_a_long_dress",
    "tank_jeans":   "a_woman_wearing_a_tank_top_and_long_jeans",
    "jacket_skirt": "a_woman_wearing_a_jacket_and_a_short_skirt",
}

# 驅動姿勢來源：IMG_4640 的某一幀（張臂姿勢，與圖4-6一致）
POSE_NPZ = "/home/jiaro/projects/romp_test/img4640_output/00000070.npz"

OUT_DIR = "outfit_change_figure"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 共用：模型、bind、姿勢 ---
model = build_smplx_model(SMPLX_DIR)
bind = load_bind_params(BIND_PKL)
G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)

source_mesh = trimesh.load(SOURCE_OBJ, force="mesh", process=False)
source_v = np.asarray(source_mesh.vertices, np.float32)

# 讀取驅動姿勢
pose_data = np.load(POSE_NPZ, allow_pickle=True)
results = pose_data["results"].item() if "results" in pose_data.files else pose_data
romp_theta = results["smpl_thetas"][0]

def find_mesh(keyword):
    pat = f"results/renderpeople/*/{SUBJECT}/*{keyword}*/mesh_refine.obj"
    cands = glob.glob(pat)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)   # 取最新

def transfer_weights(chupa_v):
    tree = cKDTree(source_v)
    _, idx = tree.query(chupa_v, k=1)
    w = lbs_w[idx]
    s = w.sum(1, keepdims=True); s[s==0]=1
    return (w/s).astype(np.float32)

for label, keyword in OUTFITS.items():
    mesh_path = find_mesh(keyword)
    if mesh_path is None:
        print(f"[跳過] {label}: 找不到 mesh（keyword={keyword}），請確認已完整生成")
        continue
    print(f"[{label}] 使用 {mesh_path}")

    m = trimesh.load(mesh_path, force="mesh", process=False)
    chupa_v = np.asarray(m.vertices, np.float32)
    faces = np.asarray(m.faces, np.int64)

    weights = transfer_weights(chupa_v)
    v_bind_raw = world_to_raw(chupa_v, bind["body_scale"], bind["translation"])

    # 綁定姿勢版本（A-pose，供對照）
    trimesh.Trimesh(vertices=chupa_v, faces=faces, process=False).export(
        f"{OUT_DIR}/{label}_apose.obj")

    # 驅動到張臂姿勢
    v_driven = drive_chupa_with_romp_frame(model, bind, weights, v_bind_raw, G_bind, romp_theta)
    trimesh.Trimesh(vertices=v_driven, faces=faces, process=False).export(
        f"{OUT_DIR}/{label}_driven.obj")
    print(f"  -> 已匯出 {label}_apose.obj 與 {label}_driven.obj")

print(f"\n[DONE] 素材於 {OUT_DIR}/")
print("  圖4-7 建議用 *_driven.obj（同姿勢、不同服裝，並排截圖）")
