from pathlib import Path
import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial import cKDTree

# ====== 1. 路徑設定 ======
smplx_model_dir = Path("src/smpl_related/models")
source_smplx_obj = Path("debug_source_smplx_rp_ben_posed_001.obj")

# 先填你要綁定的 Chupa mesh_refine.obj
chupa_mesh_path = Path(
    "results/renderpeople/20260604-143444/rp_ben_posed_001/"
    "a_person_wearing_a_red_jacket_and_black_pants_0042/mesh_refine.obj"
)

out_dir = Path("animation_data")
out_dir.mkdir(parents=True, exist_ok=True)

out_weights = out_dir / "chupa_skinning_weights.npy"
out_template = out_dir / "chupa_template_vertices.npy"
out_faces = out_dir / "chupa_faces.npy"
out_nearest = out_dir / "chupa_nearest_smplx_vertex.npy"

# ====== 2. 讀取 source SMPL-X mesh 與 Chupa mesh ======
print("[INFO] Loading source SMPL-X mesh:", source_smplx_obj)
source_mesh = trimesh.load(source_smplx_obj, force="mesh", process=False)

print("[INFO] Loading Chupa mesh:", chupa_mesh_path)
chupa_mesh = trimesh.load(chupa_mesh_path, force="mesh", process=False)

source_verts = np.asarray(source_mesh.vertices, dtype=np.float32)
chupa_verts = np.asarray(chupa_mesh.vertices, dtype=np.float32)
chupa_faces = np.asarray(chupa_mesh.faces, dtype=np.int64)

print("[INFO] source verts:", source_verts.shape)
print("[INFO] chupa verts:", chupa_verts.shape)
print("[INFO] chupa faces:", chupa_faces.shape)

# ====== 3. 建立 SMPL-X model，取得 lbs_weights ======
model = smplx.create(
    model_path=str(smplx_model_dir),
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    num_betas=10,
    num_expression_coeffs=10,
    batch_size=1,
)

# lbs_weights: (10475, 55) 或類似維度，代表每個 SMPL-X vertex 對每個 joint 的權重
lbs_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
print("[INFO] SMPL-X lbs_weights:", lbs_weights.shape)

if lbs_weights.shape[0] != source_verts.shape[0]:
    raise ValueError(
        f"SMPL-X lbs_weights vertex count {lbs_weights.shape[0]} "
        f"!= source mesh vertex count {source_verts.shape[0]}"
    )

# ====== 4. 最近頂點對應：Chupa vertex -> nearest SMPL-X vertex ======
print("[INFO] Building KDTree...")
tree = cKDTree(source_verts)

print("[INFO] Query nearest SMPL-X vertex for each Chupa vertex...")
dist, nearest_idx = tree.query(chupa_verts, k=1)

# ====== 診斷 E：插入在這裡 ======
print("\n=== 診斷 E ===")
print("nearest_idx dtype:", nearest_idx.dtype)
print("nearest_idx shape:", nearest_idx.shape)
print("nearest_idx 範圍:", nearest_idx.min(), "~", nearest_idx.max())
print("lbs_weights 的頂點數(合法 index 上限):", lbs_weights.shape[0])

left_elbow_range = np.arange(4176, 4200)
left_finger_range = np.arange(4641, 4700)
print("nearest_idx 裡落在左手肘範圍(4176~4200)的數量:",
      np.isin(nearest_idx, left_elbow_range).sum())
print("nearest_idx 裡落在左手指範圍(4641~4700)的數量:",
      np.isin(nearest_idx, left_finger_range).sum())

# 直接取幾個實際配對到左手肘範圍的 chupa vertex，檢查它們的權重是否真的是 0
matched_chupa_idx = np.where(np.isin(nearest_idx, left_elbow_range))[0]
print("有幾個 chupa 頂點被配對到左手肘 source 範圍:", len(matched_chupa_idx))
if len(matched_chupa_idx) > 0:
    sample = matched_chupa_idx[:5]
    print("這幾個 chupa 頂點對應的 nearest_idx:", nearest_idx[sample])
    print("這些 nearest_idx 在 lbs_weights 裡的原始權重(第18欄應該>0):")
    print(lbs_weights[nearest_idx[sample]][:, [17,18,19,20,21]])
# ====== 診斷 E 結束 ======
# ====== 5. 轉移 skinning weights ======
chupa_weights = lbs_weights[nearest_idx]

# 正規化，避免數值誤差
weight_sum = chupa_weights.sum(axis=1, keepdims=True)
weight_sum[weight_sum == 0] = 1.0
chupa_weights = chupa_weights / weight_sum

print("[INFO] chupa weights:", chupa_weights.shape)
print("[INFO] weight sum min/max:", chupa_weights.sum(axis=1).min(), chupa_weights.sum(axis=1).max())

# ====== 6. 儲存結果 ======
np.save(out_weights, chupa_weights)
np.save(out_template, chupa_verts)
np.save(out_faces, chupa_faces)
np.save(out_nearest, nearest_idx)

print("[DONE] saved:")
print(" ", out_weights)
print(" ", out_template)
print(" ", out_faces)
print(" ", out_nearest)
