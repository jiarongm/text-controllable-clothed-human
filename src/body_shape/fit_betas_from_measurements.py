"""
fit_betas_from_measurements.py
從使用者輸入的身高(cm)與體重(kg)，最佳化求出對應的 SMPL-X betas。
在 chupa 環境執行。
"""
import numpy as np
import torch
import trimesh
import smplx

SMPLX_DIR = "src/smpl_related/models"
HUMAN_DENSITY = 1000.0   # kg/m^3，人體密度約等於水

def build_model():
    return smplx.create(SMPLX_DIR, model_type="smplx", gender="neutral",
                        use_pca=True, num_pca_comps=6, num_betas=10,
                        num_expression_coeffs=10, batch_size=1)

def mesh_height(verts):
    return verts[:, 1].max() - verts[:, 1].min()

def mesh_volume(verts, faces):
    """用 trimesh 算封閉網格體積(m^3)"""
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return abs(m.volume)

def fit(target_height_cm, target_weight_kg, iters=1500, lr=0.05, w_reg=0.0005):
    model = build_model()
    faces = model.faces
    target_h = target_height_cm / 100.0
    target_v = target_weight_kg / HUMAN_DENSITY

    betas = torch.zeros(1, 10, requires_grad=True)
    opt = torch.optim.Adam([betas], lr=lr)

    print(f"目標: 身高 {target_height_cm} cm, 體重 {target_weight_kg} kg")

    f = torch.tensor(faces.astype(np.int64))
    for it in range(iters):
        opt.zero_grad()
        out = model(betas=betas)
        v = out.vertices[0]

        h = v[:, 1].max() - v[:, 1].min()
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        vol = torch.abs((v0 * torch.cross(v1, v2, dim=1)).sum() / 6.0)

        # 關鍵：大幅放大身高與體積損失的權重
        loss_h = ((h - target_h) ** 2) * 1000.0
        loss_v = ((vol - target_v) ** 2) * 5000.0
        loss_reg = (betas ** 2).sum() * w_reg
        loss = loss_h + loss_v + loss_reg

        loss.backward()
        opt.step()

        if it % 300 == 0:
            print(f"  iter {it}: 身高 {h.item()*100:.1f} cm, "
                  f"體重 {vol.item()*HUMAN_DENSITY:.1f} kg, loss {loss.item():.4f}")

    with torch.no_grad():
        out = model(betas=betas)
        v = out.vertices[0].numpy()
    h_final = mesh_height(v) * 100
    w_final = mesh_volume(v, faces) * HUMAN_DENSITY
    print(f"\n結果: 身高 {h_final:.1f} cm (目標 {target_height_cm}, "
          f"誤差 {abs(h_final-target_height_cm):.1f}), "
          f"體重 {w_final:.1f} kg (目標 {target_weight_kg}, "
          f"誤差 {abs(w_final-target_weight_kg):.1f})")
    print(f"betas: {betas.detach().numpy().reshape(-1).round(3)}")
    return betas.detach().numpy().reshape(-1), v, faces
    
if __name__ == "__main__":
    # 測試幾組
    import os
    os.makedirs("measurement_fit", exist_ok=True)
    for name, (h, w) in {
        "tall_thin":  (180, 60),
        "average":    (165, 60),
        "short_heavy":(155, 75),
    }.items():
        print(f"\n=== {name} ===")
        betas, verts, faces = fit(h, w)
        trimesh.Trimesh(verts, faces, process=False).export(f"measurement_fit/{name}.obj")
        np.save(f"measurement_fit/{name}_betas.npy", betas)
    print("\n[DONE] 網格與 betas 已存至 measurement_fit/")
