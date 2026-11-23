import pickle
import numpy as np
import torch
import smplx
import trimesh

# ====== 1. 讀取 bind pose 參數(檢查6/7用的檔案) ======
with open("data/renderpeople/smplx/rp_ben_posed_001.pkl", "rb") as f:
    bind_params = pickle.load(f)

bind_global_orient = np.array(bind_params["global_orient"]).reshape(1, 3).astype(np.float32)
bind_betas = np.array(bind_params["betas"]).reshape(1, 10).astype(np.float32)
bp66 = np.array(bind_params["body_pose"]).reshape(22, 3).astype(np.float32)

# ====== 2. 讀取 source SMPL-X mesh(檢查2/8用的檔案) ======
source_mesh = trimesh.load(
    "debug_source_smplx_rp_ben_posed_001.obj", force="mesh", process=False
)
source_verts = np.asarray(source_mesh.vertices, dtype=np.float32)
print("source_verts shape:", source_verts.shape)

# ====== 3. 建立 SMPL-X model(檢查8用的) ======
model = smplx.create(
    "src/smpl_related/models",
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    num_betas=10,
    batch_size=1,
)

# ====== 4. 檢查10:窮舉22種丟棄組合，找出正確的body_pose切法 ======
results = []
for drop_idx in range(22):
    keep_mask = np.ones(22, dtype=bool)
    keep_mask[drop_idx] = False
    bp63 = bp66[keep_mask].reshape(1, 63)

    out = model(
        global_orient=torch.tensor(bind_global_orient, dtype=torch.float32),
        body_pose=torch.tensor(bp63, dtype=torch.float32),
        betas=torch.tensor(bind_betas, dtype=torch.float32),
        return_verts=True,
    )
    verts_check = out.vertices.detach().numpy()[0]
    diff = np.linalg.norm(verts_check - source_verts, axis=1)
    results.append((drop_idx, float(diff.mean()), float(diff.max())))

results.sort(key=lambda x: x[1])
print("\n=== 依 mean diff 排序(最小的在最上面) ===")
for drop_idx, mean_d, max_d in results:
    print(f"drop slot {drop_idx}: mean diff={mean_d:.6f}, max diff={max_d:.6f}")
