"""
build_shape_assets.py

為每個身形預先計算並儲存完整的「綁定資產」，供 Gradio 即時切換使用。
每組資產包含：頂點、蒙皮權重、面片、G_bind、以及身高體重等 metadata。

在 chupa 環境、chupa 專案目錄執行。

輸出結構：
  shape_assets/
  ├── tall_thin/
  │   ├── template_vertices.npy   (v_bind_world)
  │   ├── weights.npy
  │   ├── faces.npy
  │   ├── G_bind.npy
  │   └── meta.json               (身高、體重、betas、subject、prompt)
  ├── average/
  └── short_heavy/
"""
import glob
import json
import os
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from chupa_reskin import (
    build_smplx_model, compute_joint_transforms, load_bind_params,
)

SMPLX_DIR = "src/smpl_related/models"
OUT_ROOT = "shape_assets"
PROMPT_KEY = "a_woman_wearing_a_long_dress"
PROMPT_TEXT = "a woman wearing a long dress"

# 身形定義：資產名稱 -> (subject, 身高cm, 體重kg)
# 把原本寫死的 SHAPES 換成從 json 讀
import json
with open("shape_library_index.json", encoding="utf-8") as f:
    lib = json.load(f)
SHAPES = {
    f"{s['height_cm']}_{s['weight_kg']}": (s["subject"], s["height_cm"], s["weight_kg"])
    for s in lib
}

model = build_smplx_model(SMPLX_DIR)
lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)


def make_source_mesh(bind):
    """用該身形的 betas + bind pose 生成 SMPL-X 裸體 source（世界座標）"""
    betas_t = torch.tensor(bind["betas"].reshape(1, -1), dtype=torch.float32)
    fp_t = torch.tensor(bind["full_pose"].reshape(1, 55, 3), dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=betas_t,
                    global_orient=fp_t[:, 0],
                    body_pose=fp_t[:, 1:22].reshape(1, 63))
    v = out.vertices[0].numpy()
    return v * bind["body_scale"] + bind["translation"].reshape(1, 3)


def transfer_weights(chupa_v, source_v):
    tree = cKDTree(source_v)
    dist, idx = tree.query(chupa_v, k=1)
    w = lbs_w[idx]
    s = w.sum(1, keepdims=True); s[s == 0] = 1
    return (w / s).astype(np.float32), dist


def find_mesh(subject):
    pat = f"results/renderpeople/*/{subject}/*{PROMPT_KEY}*/mesh_refine.obj"
    cands = glob.glob(pat)
    return max(cands, key=os.path.getmtime) if cands else None


os.makedirs(OUT_ROOT, exist_ok=True)

for asset_name, (subject, height_cm, weight_kg) in SHAPES.items():
    mesh_path = find_mesh(subject)
    if mesh_path is None:
        print(f"[跳過] {asset_name}: 找不到 {subject} 的著衣網格")
        continue

    print(f"\n[{asset_name}] {height_cm}cm / {weight_kg}kg")
    print(f"  網格: {mesh_path}")

    bind = load_bind_params(f"data/renderpeople/smplx/{subject}.pkl")
    G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
    source_v = make_source_mesh(bind)

    m = trimesh.load(mesh_path, force="mesh", process=False)
    chupa_v = np.asarray(m.vertices, np.float32)
    faces = np.asarray(m.faces, np.int64)

    weights, dist = transfer_weights(chupa_v, source_v)
    print(f"  權重轉移: nearest dist mean={dist.mean():.4f} max={dist.max():.4f}")

    # 健康度檢查
    if np.isnan(weights).any():
        print("  [錯誤] 權重含 NaN，跳過")
        continue
    if not (np.abs(weights.sum(1) - 1.0) < 1e-3).all():
        print("  [錯誤] 權重列和不為 1，跳過")
        continue

    out_dir = os.path.join(OUT_ROOT, asset_name)
    os.makedirs(out_dir, exist_ok=True)

    np.save(f"{out_dir}/template_vertices.npy", chupa_v)
    np.save(f"{out_dir}/weights.npy", weights)
    np.save(f"{out_dir}/faces.npy", faces)
    np.save(f"{out_dir}/G_bind.npy", G_bind)

    meta = {
        "asset_name": asset_name,
        "subject": subject,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "betas": bind["betas"].reshape(-1).tolist(),
        "body_scale": float(bind["body_scale"]),
        "translation": bind["translation"].reshape(-1).tolist(),
        "prompt": PROMPT_TEXT,
        "num_vertices": int(len(chupa_v)),
    }
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  已存至 {out_dir}/ (頂點數 {len(chupa_v)})")

print(f"\n[DONE] 資產已建立於 {OUT_ROOT}/")
print("      Gradio 可讀取各資料夾的 meta.json，依身高體重找最接近者載入")
