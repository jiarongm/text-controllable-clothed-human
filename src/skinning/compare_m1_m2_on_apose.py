"""
compare_m1_m2_on_apose.py

在實際使用的 emily A-pose 資產（無自我接觸）上，比較 M1 與 M2：
  1. 權重建立時間（離線一次性成本）
  2. 驅動到同一姿勢，輸出網格供視覺比對「乾淨資產上兩者是否有實質差異」

在 chupa 環境執行。
"""
import time
import numpy as np
import trimesh
import smplx
from scipy.spatial import cKDTree

from romp_to_chupa_realtime import drive_chupa_with_romp_frame
from chupa_reskin import (
    build_smplx_model, compute_joint_transforms, load_bind_params, world_to_raw,
)

# --- emily A-pose 資產 ---
SUBJECT = "rp_emily_apose_001"
SOURCE_OBJ = "debug_source_smplx_rp_emily_apose_001.obj"
BIND_PKL = "data/renderpeople/smplx/rp_emily_apose_001.pkl"
SMPLX_DIR = "src/smpl_related/models"
# 用長裙那件已生成的網格當測試對象（換成你要的 mesh 路徑）
CHUPA_MESH = None  # None 則自動找最新的 long_dress mesh
DRIVE_POSE_NPZ = "/home/jiaro/projects/romp_test/img4640_output/00000070.npz"

OUT_DIR = "m1_m2_apose_compare"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def find_latest_mesh():
    import glob
    pat = f"results/renderpeople/*/{SUBJECT}/*long_dress*/mesh_refine.obj"
    cands = glob.glob(pat)
    return max(cands, key=os.path.getmtime) if cands else None


def normalize_rows(w):
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (w / s).astype(np.float32)


def build_M1(chupa_v, source_v, lbs_w):
    tree = cKDTree(source_v)
    _, idx = tree.query(chupa_v, k=1)
    return normalize_rows(lbs_w[idx])


def build_M2(chupa_v, source_v, lbs_w, k=12):
    tree = cKDTree(source_v)
    dist, idx = tree.query(chupa_v, k=k)
    sigma = np.median(dist[:, 1]) + 1e-8
    gw = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    gw /= gw.sum(axis=1, keepdims=True)
    w = np.einsum("vk,vkj->vj", gw, lbs_w[idx])
    return normalize_rows(w)


def main():
    mesh_path = CHUPA_MESH or find_latest_mesh()
    if mesh_path is None:
        print("[錯誤] 找不到測試網格，請設定 CHUPA_MESH")
        return
    print(f"[INFO] 測試網格: {mesh_path}")

    source_v = np.asarray(trimesh.load(SOURCE_OBJ, force="mesh", process=False).vertices, np.float32)
    m = trimesh.load(mesh_path, force="mesh", process=False)
    chupa_v = np.asarray(m.vertices, np.float32)
    faces = np.asarray(m.faces, np.int64)

    model = build_smplx_model(SMPLX_DIR)
    bind = load_bind_params(BIND_PKL)
    G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])
    lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)

    # ---- 1. 權重建立時間（各跑 5 次取平均，排除快取影響）----
    print("\n=== 權重建立時間（離線一次性成本）===")
    for name, fn in [("M1", lambda: build_M1(chupa_v, source_v, lbs_w)),
                     ("M2", lambda: build_M2(chupa_v, source_v, lbs_w))]:
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            w = fn()
            ts.append(time.perf_counter() - t0)
        print(f"  {name}: {np.mean(ts)*1000:.1f} ms (平均, n=5, 頂點數={len(chupa_v)})")

    # ---- 2. 驅動到同姿勢，輸出網格供視覺比對 ----
    print("\n=== 驅動輸出（供視覺比對）===")
    pose_data = np.load(DRIVE_POSE_NPZ, allow_pickle=True)
    results = pose_data["results"].item() if "results" in pose_data.files else pose_data
    romp_theta = results["smpl_thetas"][0]
    v_bind_raw = world_to_raw(chupa_v, bind["body_scale"], bind["translation"])

    W = {"M1": build_M1(chupa_v, source_v, lbs_w),
         "M2": build_M2(chupa_v, source_v, lbs_w)}

    driven = {}
    for name, weights in W.items():
        v_driven = drive_chupa_with_romp_frame(model, bind, weights, v_bind_raw, G_bind, romp_theta)
        driven[name] = v_driven
        trimesh.Trimesh(vertices=v_driven, faces=faces, process=False).export(
            f"{OUT_DIR}/driven_{name}.obj")
        print(f"  已輸出 {OUT_DIR}/driven_{name}.obj")

    # ---- 3. 量化兩者驅動結果的差異（頂點位移差）----
    diff = np.linalg.norm(driven["M1"] - driven["M2"], axis=1)
    bbox_diag = np.linalg.norm(chupa_v.max(0) - chupa_v.min(0))
    print("\n=== M1 vs M2 驅動結果差異（乾淨 A-pose 資產）===")
    print(f"  頂點位移差 平均: {diff.mean()*1000:.3f} mm")
    print(f"  頂點位移差 最大: {diff.max()*1000:.3f} mm")
    print(f"  相對於身高(bbox對角{bbox_diag:.2f}m)的平均差異: {100*diff.mean()/bbox_diag:.4f}%")
    print(f"\n[DONE] 請將 {OUT_DIR}/driven_M1.obj 與 driven_M2.obj 匯入 Blender 並排比對")


if __name__ == "__main__":
    main()
