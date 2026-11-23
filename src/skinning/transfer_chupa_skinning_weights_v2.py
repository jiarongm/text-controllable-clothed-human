from pathlib import Path
import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial import cKDTree

# ====== 路徑設定 ======
smplx_model_dir = Path("src/smpl_related/models")
source_smplx_obj = Path("debug_source_smplx_rp_emily_apose_001.obj")

SUBJECT = "rp_emily_apose_001"
out_dir = Path("animation_data_emily_apose")
out_dir.mkdir(parents=True, exist_ok=True)

out_weights = out_dir / "chupa_skinning_weights.npy"
out_template = out_dir / "chupa_template_vertices.npy"
out_faces = out_dir / "chupa_faces.npy"
out_nearest = out_dir / "chupa_nearest_smplx_vertex.npy"

# ====== 自動找出最新一次生成的 mesh_refine.obj ======
candidates = sorted(
    Path("results/renderpeople").glob(f"*/{SUBJECT}/*/mesh_refine.obj"),
    key=lambda p: p.stat().st_mtime,
)
if not candidates:
    raise FileNotFoundError(
        f"找不到任何 {SUBJECT} 的 mesh_refine.obj，請確認生成是否完成"
    )

chupa_mesh_path = candidates[-1]
print(f"[INFO] 找到 {len(candidates)} 個候選，使用最新的:")
print("      ", chupa_mesh_path)

# ====== 讀取 source SMPL-X mesh 與 Chupa mesh ======
print("[INFO] Loading source SMPL-X mesh:", source_smplx_obj)
if not source_smplx_obj.exists():
    raise FileNotFoundError(f"{source_smplx_obj} 不存在，請先跑 export_chupa_source_smplx_v2.py")
source_mesh = trimesh.load(source_smplx_obj, force="mesh", process=False)

print("[INFO] Loading Chupa mesh:", chupa_mesh_path)
chupa_mesh = trimesh.load(chupa_mesh_path, force="mesh", process=False)

source_verts = np.asarray(source_mesh.vertices, dtype=np.float32)
chupa_verts = np.asarray(chupa_mesh.vertices, dtype=np.float32)
chupa_faces = np.asarray(chupa_mesh.faces, dtype=np.int64)

print("[INFO] source verts:", source_verts.shape)
print("[INFO] chupa verts:", chupa_verts.shape)
print("[INFO] chupa faces:", chupa_faces.shape)

# ====== 地基驗證：source mesh 與 nds 初始網格的 size 是否一致 ======
nds_initial = chupa_mesh_path.parent / "nds_dual" / "meshes" / "2views_0001.obj"
if nds_initial.exists():
    early = trimesh.load(nds_initial, force="mesh", process=False)
    early_verts = np.asarray(early.vertices)
    early_size = early_verts.max(0) - early_verts.min(0)
    source_size = source_verts.max(0) - source_verts.min(0)
    ratio = source_size / early_size
    print("\n=== 地基驗證（source vs nds初始網格）===")
    print("source size:", source_size)
    print("nds    size:", early_size)
    print("三軸比例   :", ratio)
    if np.any(np.abs(ratio - 1.0) > 0.05):
        raise ValueError(
            "三軸比例偏離 1.0 超過 5%，代表 source mesh 與 Chupa mesh 座標系不一致，"
            "請先確認 export_chupa_source_smplx_v2.py 讀取的是正確的 .pkl"
        )
    print("[OK] 地基驗證通過\n")
else:
    print("[WARN] 找不到 2views_0001.obj，略過地基驗證")

# ====== 建立 SMPL-X model 取得 lbs_weights ======
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

lbs_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
print("[INFO] SMPL-X lbs_weights:", lbs_weights.shape)

if lbs_weights.shape[0] != source_verts.shape[0]:
    raise ValueError(
        f"lbs_weights vertex count {lbs_weights.shape[0]} "
        f"!= source verts {source_verts.shape[0]}"
    )

# ====== 最近頂點權重轉移 ======
print("[INFO] Building KDTree...")
tree = cKDTree(source_verts)
dist, nearest_idx = tree.query(chupa_verts, k=1)

print("[INFO] nearest distance: min={:.4f} mean={:.4f} max={:.4f}".format(
    dist.min(), dist.mean(), dist.max()))

chupa_weights = lbs_weights[nearest_idx]
weight_sum = chupa_weights.sum(axis=1, keepdims=True)
weight_sum[weight_sum == 0] = 1.0
chupa_weights = (chupa_weights / weight_sum).astype(np.float32)

print("[INFO] chupa weights:", chupa_weights.shape)
row_sum = chupa_weights.sum(axis=1)
print("[INFO] row sum min/max:", row_sum.min(), row_sum.max())
print("[INFO] NaN count:", np.isnan(chupa_weights).sum())

# ====== 權重健康度驗證 ======
col_sum = chupa_weights.sum(axis=0)

print("\n=== 左手臂鏈驗證 (18=elbow, 20=wrist, 25~39=手指) ===")
for j in [18, 20] + list(range(25, 40)):
    nonzero = (chupa_weights[:, j] > 1e-6).sum()
    print(f"joint {j}: col_sum={col_sum[j]:.4f}, 非零頂點數={nonzero}")

print("\n=== 右手臂鏈驗證 (19=elbow, 21=wrist, 40~54=手指) ===")
for j in [19, 21] + list(range(40, 55)):
    nonzero = (chupa_weights[:, j] > 1e-6).sum()
    print(f"joint {j}: col_sum={col_sum[j]:.4f}, 非零頂點數={nonzero}")

zero_cols = np.where(col_sum < 1e-6)[0]
if len(zero_cols) > 0:
    print("\n[WARN] 以下關節完全沒有頂點綁定，若包含主要肢體關節請回頭檢查:")
    print("      ", zero_cols.tolist())
else:
    print("\n[OK] 所有 55 個關節都有頂點綁定")

# ====== 儲存 ======
np.save(out_weights, chupa_weights)
np.save(out_template, chupa_verts)
np.save(out_faces, chupa_faces)
np.save(out_nearest, nearest_idx)

print("\n[DONE] saved to", out_dir)
for p in [out_weights, out_template, out_faces, out_nearest]:
    print("  ", p)