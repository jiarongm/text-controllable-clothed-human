"""
smpl_betas_to_smplx.py
將 ROMP 的 SMPL betas 轉換為 SMPL-X betas（方案一：關節位置擬合）。
先驗證：輸出 SMPL 目標網格與擬合後的 SMPL-X 網格，肉眼比對體型是否接近。
在 chupa 環境執行。
"""
import numpy as np
import torch
import trimesh
import smplx

SMPL_DIR = "src/smpl_related/models"
ROMP_NPZ = "/home/jiaro/projects/romp_test/img4640_output/00000070.npz"
OUT_DIR = "smpl2smplx_verify"
import os; os.makedirs(OUT_DIR, exist_ok=True)

device = "cpu"

# --- 讀 ROMP 的 SMPL betas ---
d = np.load(ROMP_NPZ, allow_pickle=True)["results"].item()
smpl_betas = torch.tensor(np.array(d["smpl_betas"]), dtype=torch.float32, device=device)  # (1,10)
print(f"ROMP SMPL betas: {smpl_betas.numpy().round(3)}")

# --- 建立 SMPL 與 SMPL-X 模型（都用 neutral，零姿態）---
smpl = smplx.create(SMPL_DIR, model_type="smpl", gender="neutral",
                    num_betas=10, batch_size=1).to(device).eval()
smplx_model = smplx.create(SMPL_DIR, model_type="smplx", gender="neutral",
                           use_pca=True, num_pca_comps=6, num_betas=10,
                           num_expression_coeffs=10, batch_size=1).to(device).eval()

# --- SMPL 目標網格與關節（零姿態）---
with torch.no_grad():
    smpl_out = smpl(betas=smpl_betas)
smpl_verts = smpl_out.vertices[0].cpu().numpy()
smpl_joints = smpl_out.joints  # (1, J, 3)
trimesh.Trimesh(smpl.v_template.numpy()*0 + smpl_verts, smpl.faces, process=False).export(
    f"{OUT_DIR}/target_smpl.obj")

# --- 待最佳化的 SMPL-X betas ---
smplx_betas = torch.zeros(1, 10, requires_grad=True, device=device)
optimizer = torch.optim.Adam([smplx_betas], lr=0.05)

# SMPL 與 SMPL-X 前 22 個 body joint 有對應（0..21）
N_COMMON = 22

print("\n開始擬合（以前 22 個關節位置為目標）...")
for it in range(300):
    optimizer.zero_grad()
    out = smplx_model(betas=smplx_betas)
    smplx_joints = out.joints  # (1, J, 3)
    # 以骨盆(0)為原點對齊，只比體型不比全域位置
    sj = smplx_joints[:, :N_COMMON] - smplx_joints[:, [0]]
    tj = smpl_joints[:, :N_COMMON] - smpl_joints[:, [0]]
    loss = ((sj - tj) ** 2).sum()
    loss.backward()
    optimizer.step()
    if it % 50 == 0:
        print(f"  iter {it}: loss {loss.item():.6f}")

print(f"\n擬合後 SMPL-X betas: {smplx_betas.detach().numpy().round(3)}")

# --- 輸出擬合後的 SMPL-X 網格 ---
with torch.no_grad():
    smplx_out = smplx_model(betas=smplx_betas)
smplx_verts = smplx_out.vertices[0].cpu().numpy()
trimesh.Trimesh(smplx_verts, smplx_model.faces, process=False).export(
    f"{OUT_DIR}/fitted_smplx.obj")

# --- 量化：關節位置殘差 ---
with torch.no_grad():
    sj = (smplx_out.joints[:, :N_COMMON] - smplx_out.joints[:, [0]]).numpy()[0]
    tj = (smpl_joints[:, :N_COMMON] - smpl_joints[:, [0]]).numpy()[0]
per_joint = np.linalg.norm(sj - tj, axis=1)
print(f"\n關節位置殘差: 平均 {per_joint.mean()*1000:.1f} mm, 最大 {per_joint.max()*1000:.1f} mm")

# 儲存轉換後的 betas 供後續生成使用
np.save(f"{OUT_DIR}/user_smplx_betas.npy", smplx_betas.detach().numpy())
print(f"\n[DONE] SMPL-X betas 已存至 {OUT_DIR}/user_smplx_betas.npy")
print(f"       請將 target_smpl.obj 與 fitted_smplx.obj 匯入 Blender 疊合比對體型")
