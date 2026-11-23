"""
explore_gender_betas.py
掃描各個 beta 分量，找出哪些會影響性別相關的體型特徵。
在 chupa 環境執行。
"""
import numpy as np
import torch
import trimesh
import smplx
import os

SMPLX_DIR = "src/smpl_related/models"
OUT_DIR = "gender_beta_scan"
os.makedirs(OUT_DIR, exist_ok=True)

model = smplx.create(SMPLX_DIR, model_type="smplx", gender="neutral",
                     use_pca=True, num_pca_comps=6, num_betas=10).eval()

def measure(v):
    """量幾個性別相關的指標"""
    y = v[:, 1]
    h = y.max() - y.min()
    # 用高度比例定位身體區段（粗略）
    y_norm = (y - y.min()) / h
    def width_at(lo, hi):
        m = (y_norm >= lo) & (y_norm <= hi)
        return v[m, 0].max() - v[m, 0].min() if m.sum() > 0 else 0
    chest_w = width_at(0.68, 0.76)    # 胸部區段寬度
    waist_w = width_at(0.58, 0.64)    # 腰部
    hip_w   = width_at(0.48, 0.55)    # 臀部
    # 胸部前後深度（Z軸），女性通常較大
    m = (y_norm >= 0.68) & (y_norm <= 0.76)
    chest_d = v[m, 2].max() - v[m, 2].min() if m.sum() > 0 else 0
    return h, chest_w, waist_w, hip_w, chest_d

print(f"{'beta':>6} {'值':>6} {'身高':>7} {'胸寬':>7} {'腰寬':>7} {'臀寬':>7} {'胸深':>7} {'臀/腰':>7}")
for i in range(10):
    for val in [-2.0, 0.0, 2.0]:
        b = np.zeros((1, 10), np.float32)
        b[0, i] = val
        with torch.no_grad():
            v = model(betas=torch.tensor(b)).vertices[0].numpy()
        h, cw, ww, hw, cd = measure(v)
        ratio = hw / ww if ww > 0 else 0
        print(f"{i:>6} {val:>6.1f} {h:>7.3f} {cw:>7.3f} {ww:>7.3f} "
              f"{hw:>7.3f} {cd:>7.3f} {ratio:>7.3f}")
        if val != 0.0:
            trimesh.Trimesh(v, model.faces, process=False).export(
                f"{OUT_DIR}/beta{i}_{val:+.0f}.obj")
    print()

print(f"\n網格已存至 {OUT_DIR}/，可匯入 Blender 目視比對")
print("找『胸深增加、臀/腰比增大』的 beta → 偏女性特徵")
print("找『肩寬增加、臀/腰比減小』的 beta → 偏男性特徵")
