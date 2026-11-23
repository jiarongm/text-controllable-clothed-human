"""
compare_skinning_methods.py

在同一個「困難測試案例」（手插口袋、具自我接觸的 rp_ben_posed_001）上，
比較四種蒙皮權重轉移方法，輸出量化指標與可供肉眼比對的驅動網格。

方法：
  M1  nearest-vertex (k=1)               ← 現行系統，基準線
  M2  k-nearest 高斯加權 (k=12)
  M3  barycentric（最近三角面重心座標插值）
  M4  body-part 約束（限制對應同部位）

指標：
  1. 相鄰頂點關節衝突率         → 蒙皮空間連續性（撕裂風險）
  2. 自我接觸區域洩漏（分桶位移）→ 特定困難情境的錯配
  3. 動作驅動後的網格粗糙度     → 實際驅動品質

用法：
  conda activate chupa
  cd ~/projects/chupa
  python compare_skinning_methods.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import trimesh
import smplx
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_scipy
from smplx.lbs import batch_rodrigues, vertices2joints, blend_shapes, batch_rigid_transform


# ============================================================
# 路徑設定
# ============================================================

SMPLX_MODEL_DIR = "src/smpl_related/models"
SOURCE_OBJ = "debug_source_smplx_rp_ben_posed_001_v2.obj"
BIND_PKL = "data/renderpeople/smplx/rp_ben_posed_001.pkl"
CHUPA_MESH = ("results/renderpeople/20260604-143444/rp_ben_posed_001/"
              "a_person_wearing_a_red_jacket_and_black_pants_0042/mesh_refine.obj")

OUT_DIR = Path("skinning_comparison")
OUT_DIR.mkdir(exist_ok=True)

NUM_JOINTS = 55
NUM_BODY_JOINTS = 21
LEFT_ELBOW_FULL_IDX = 18

# 55 關節歸併為大部位（body-part 約束與替代分割用）
# 依 SMPL-X 關節語意粗分：軀幹/頭、左臂、右臂、左腿、右腿
PART_OF_JOINT = {}
for j in range(55):
    PART_OF_JOINT[j] = "torso"
for j in [16, 18, 20] + list(range(25, 40)):   # 左肩肘腕 + 左手指
    PART_OF_JOINT[j] = "left_arm"
for j in [17, 19, 21] + list(range(40, 55)):   # 右肩肘腕 + 右手指
    PART_OF_JOINT[j] = "right_arm"
for j in [1, 4, 7, 10]:                          # 左髖膝踝趾
    PART_OF_JOINT[j] = "left_leg"
for j in [2, 5, 8, 11]:                          # 右髖膝踝趾
    PART_OF_JOINT[j] = "right_leg"


# ============================================================
# SMPL-X forward kinematics（與主系統一致）
# ============================================================

def build_model():
    m = smplx.create(SMPLX_MODEL_DIR, model_type="smplx", gender="neutral",
                     use_pca=True, num_pca_comps=6, num_betas=10,
                     num_expression_coeffs=10, batch_size=1)
    m.eval()
    return m


def make_full_pose(global_orient_3, body_pose_63):
    fp = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    fp[0] = global_orient_3.reshape(3)
    fp[1:1 + NUM_BODY_JOINTS] = body_pose_63.reshape(NUM_BODY_JOINTS, 3)
    return fp


def compute_G(model, betas, full_pose):
    betas_t = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    fp_t = torch.tensor(full_pose.reshape(1, NUM_JOINTS, 3), dtype=torch.float32)
    v_template = model.v_template.unsqueeze(0)
    shapedirs = model.shapedirs[:, :, :betas_t.shape[-1]]
    v_shaped = v_template + blend_shapes(betas_t, shapedirs)
    J_rest = vertices2joints(model.J_regressor, v_shaped)
    rot = batch_rodrigues(fp_t.view(-1, 3)).view(1, NUM_JOINTS, 3, 3)
    _, A = batch_rigid_transform(rot, J_rest, model.parents)
    return A.squeeze(0).detach().numpy()


def reskin(v_bind, weights, G_bind, G_new):
    V = v_bind.shape[0]
    T = np.matmul(G_new, np.linalg.inv(G_bind))
    blended = np.einsum("vj,jab->vab", weights, T)
    vh = np.concatenate([v_bind, np.ones((V, 1), np.float32)], axis=1)
    return np.einsum("vab,vb->va", blended, vh)[:, :3]


def load_bind(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    go = np.array(p["global_orient"], np.float32).reshape(3)
    betas = np.array(p["betas"], np.float32).reshape(-1)
    bp66 = np.array(p["body_pose"], np.float32).reshape(22, 3)
    bp63 = bp66[1:22].reshape(63)
    scale = float(np.asarray(p["body_scale"]).reshape(-1)[0])
    transl = np.asarray(p["global_body_translation"], np.float32).reshape(3)
    return {"betas": betas, "full_pose": make_full_pose(go, bp63),
            "scale": scale, "transl": transl}


# ============================================================
# 四種權重轉移方法
# ============================================================

def normalize_rows(w):
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (w / s).astype(np.float32)


def transfer_M1_nearest(chupa_v, source_v, lbs_w):
    tree = cKDTree(source_v)
    _, idx = tree.query(chupa_v, k=1)
    return normalize_rows(lbs_w[idx]), idx


def transfer_M2_knn_gaussian(chupa_v, source_v, lbs_w, k=12):
    tree = cKDTree(source_v)
    dist, idx = tree.query(chupa_v, k=k)            # (V, k)
    sigma = np.median(dist[:, 1]) + 1e-8            # 以第二近鄰的中位距離為尺度
    gw = np.exp(-(dist ** 2) / (2 * sigma ** 2))    # (V, k) 高斯權重
    gw /= gw.sum(axis=1, keepdims=True)
    w = np.einsum("vk,vkj->vj", gw, lbs_w[idx])     # 加權平均鄰近權重
    return normalize_rows(w), idx[:, 0]


def transfer_M3_barycentric(chupa_v, source_mesh, lbs_w):
    # 投影每個 chupa 頂點到 source 表面最近點，取該三角面三頂點權重的重心座標插值
    prox = trimesh.proximity.ProximityQuery(source_mesh)
    closest, dist, tri_id = prox.on_surface(chupa_v)
    faces = np.asarray(source_mesh.faces)
    tri_verts = np.asarray(source_mesh.vertices)[faces[tri_id]]   # (V,3,3)

    # 計算重心座標
    v0 = tri_verts[:, 1] - tri_verts[:, 0]
    v1 = tri_verts[:, 2] - tri_verts[:, 0]
    v2 = closest - tri_verts[:, 0]
    d00 = np.einsum("vi,vi->v", v0, v0)
    d01 = np.einsum("vi,vi->v", v0, v1)
    d11 = np.einsum("vi,vi->v", v1, v1)
    d20 = np.einsum("vi,vi->v", v2, v0)
    d21 = np.einsum("vi,vi->v", v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-12
    v = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w2
    bary = np.stack([u, v, w2], axis=1)             # (V,3)
    bary = np.clip(bary, 0, None)
    bary /= bary.sum(axis=1, keepdims=True)

    w = np.einsum("vt,vtj->vj", bary, lbs_w[faces[tri_id]])   # 插值三頂點權重
    idx = faces[tri_id][:, 0]
    return normalize_rows(w), idx


def transfer_M4_bodypart(chupa_v, source_v, lbs_w, source_part_of_vertex, k_vote=12):
    """
    body-part 約束：
    1. 用 k 近鄰「多數決」決定每個 chupa 頂點的目標部位
       （關鍵：用鄰域共識，而非單一最近點，否則約束會退化成 M1）
    2. 限制只在該部位的來源頂點中找最近點轉移權重
    """
    # 步驟 1：k 近鄰多數決決定部位
    tree_all = cKDTree(source_v)
    _, idx_k = tree_all.query(chupa_v, k=k_vote)          # (V, k_vote)

    part_names = sorted(set(source_part_of_vertex))
    p2i = {p: i for i, p in enumerate(part_names)}
    source_part_int = np.array([p2i[p] for p in source_part_of_vertex])

    neighbor_parts = source_part_int[idx_k]              # (V, k_vote)
    chupa_part_int = np.array([
        np.bincount(row, minlength=len(part_names)).argmax()
        for row in neighbor_parts
    ])
    chupa_part = np.array([part_names[i] for i in chupa_part_int])

    # 診斷：多數決部位 vs 全域最近點部位 的不一致數（應與 diagnose_m4.py 一致）
    part_nearest = source_part_of_vertex[idx_k[:, 0]]
    print(f"    [M4] 多數決 vs 最近點 不一致: {(chupa_part != part_nearest).sum()} / {len(chupa_v)}")

    # 步驟 2：每個部位建獨立 KDTree，限制在同部位內找最近點
    part_trees, part_vidx = {}, {}
    for part in part_names:
        vmask = np.where(source_part_of_vertex == part)[0]
        if len(vmask) == 0:
            continue
        part_vidx[part] = vmask
        part_trees[part] = cKDTree(source_v[vmask])

    idx = np.empty(len(chupa_v), dtype=np.int64)
    for part in set(chupa_part):
        cmask = np.where(chupa_part == part)[0]
        _, local = part_trees[part].query(chupa_v[cmask], k=1)
        idx[cmask] = part_vidx[part][local]

    # 診斷：限制查詢後，idx 相對全域最近點改變了多少
    idx_global = idx_k[:, 0]
    print(f"    [M4] 限制查詢後 idx 改變的頂點數: {(idx != idx_global).sum()} / {len(chupa_v)}")

    return normalize_rows(lbs_w[idx]), idx

def get_source_part_labels(source_v, lbs_w):
    # 每個 source 頂點的部位 = 其 dominant joint 所屬部位
    dom = lbs_w.argmax(axis=1)
    return np.array([PART_OF_JOINT[j] for j in dom])


# ============================================================
# 三種評估指標
# ============================================================

def metric_adjacency_conflict(weights, faces):
    """相鄰頂點 dominant joint 不同的比例（全網格）。越低越平滑。"""
    dom = weights.argmax(axis=1)
    edges = set()
    for a, b, c in faces:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))
    edges = np.array(list(edges))
    conflict = (dom[edges[:, 0]] != dom[edges[:, 1]]).sum()
    return conflict / len(edges)


def metric_selfcontact_leakage(weights, v_bind_world, G_bind, scale, transl):
    """
    自我接觸洩漏：手部主導頂點中，與軀幹/髖部頂點距離 < 1cm 的比例。
    越低代表手部權重越少滲入貼近的軀幹（錯配風險越低）。
    """
    dom = weights.argmax(axis=1)
    LEFT_WRIST, RIGHT_WRIST = 20, 21
    LEFT_FING, RIGHT_FING = list(range(25, 40)), list(range(40, 55))
    TORSO_HIP = [0, 1, 2, 3, 6, 9]

    hand_mask = np.isin(dom, [LEFT_WRIST, RIGHT_WRIST] + LEFT_FING + RIGHT_FING)
    torso_mask = np.isin(dom, TORSO_HIP)
    if hand_mask.sum() == 0 or torso_mask.sum() == 0:
        return float("nan")

    tree = cKDTree(v_bind_world[torso_mask])
    dist, _ = tree.query(v_bind_world[hand_mask], k=1)
    return (dist < 0.01).sum() / hand_mask.sum()


def metric_driven_roughness(v_driven, faces):
    """驅動後網格的拉普拉斯粗糙度（正規化，可跨方法比較）。越低越平滑。"""
    nbrs = defaultdict(set)
    for a, b, c in faces:
        nbrs[a].update([b, c]); nbrs[b].update([a, c]); nbrs[c].update([a, b])
    lap = np.zeros(len(v_driven))
    for i in range(len(v_driven)):
        if nbrs[i]:
            lap[i] = np.linalg.norm(v_driven[i] - v_driven[list(nbrs[i])].mean(axis=0))
    edges = set()
    for a, b, c in faces:
        edges.add((min(a, b), max(a, b)))
    edges = np.array(list(edges))
    edge_len = np.linalg.norm(v_driven[edges[:, 0]] - v_driven[edges[:, 1]], axis=1).mean()
    return lap.mean() / edge_len


def metric_local_tearing(v_driven, v_bind_world, faces, region_center, radius=0.15):
    """
    局部撕裂指標：只在 region_center 半徑內（口袋自我接觸區）計算。
    撕裂的本質是相鄰頂點被分到不同骨骼、驅動後往不同方向跑，
    故以「相鄰頂點位移向量的差異」衡量，而非邊長變化。
    """
    disp = v_driven - v_bind_world            # 每個頂點的位移向量
    disp_norm = np.linalg.norm(disp, axis=1)

    edges = set()
    for a, b, c in faces:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(a, c), max(a, c)))
    edges = np.array(list(edges))

    # 只取落在自我接觸區域的邊
    edge_mid = (v_bind_world[edges[:, 0]] + v_bind_world[edges[:, 1]]) / 2
    in_region = np.linalg.norm(edge_mid - region_center.reshape(1, 3), axis=1) < radius
    reg_edges = edges[in_region]

    if len(reg_edges) == 0:
        return {"region_edges": 0, "disp_discontinuity_mean": float("nan"),
                "disp_discontinuity_p95": float("nan"),
                "disp_discontinuity_max": float("nan")}

    # 相鄰頂點位移向量的差異（撕裂處會很大）
    diff = np.linalg.norm(disp[reg_edges[:, 0]] - disp[reg_edges[:, 1]], axis=1)

    # 用邊長正規化，避免長邊天生差異大
    edge_len_bind = np.linalg.norm(
        v_bind_world[reg_edges[:, 0]] - v_bind_world[reg_edges[:, 1]], axis=1)
    diff_norm = diff / (edge_len_bind + 1e-8)

    return {
        "region_edges": int(len(reg_edges)),
        "disp_discontinuity_mean": float(diff_norm.mean()),
        "disp_discontinuity_p95": float(np.percentile(diff_norm, 95)),
        "disp_discontinuity_max": float(diff_norm.max()),
    }


# ============================================================
# 主流程
# ============================================================

def main():
    print("[INFO] 載入資料...")
    source_mesh = trimesh.load(SOURCE_OBJ, force="mesh", process=False)
    chupa_mesh = trimesh.load(CHUPA_MESH, force="mesh", process=False)
    source_v = np.asarray(source_mesh.vertices, np.float32)
    chupa_v = np.asarray(chupa_mesh.vertices, np.float32)
    faces = np.asarray(chupa_mesh.faces, np.int64)

    model = build_model()
    lbs_w = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    bind = load_bind(BIND_PKL)
    G_bind = compute_G(model, bind["betas"], bind["full_pose"])
    source_part = get_source_part_labels(source_v, lbs_w)

    # 驅動測試姿勢：在 bind pose 上讓左手肘再彎 40 度
    delta = R_scipy.from_rotvec(np.deg2rad(40.0) * np.array([0, 0, 1.0]))
    new_fp = bind["full_pose"].copy()
    R_bind = R_scipy.from_rotvec(new_fp[LEFT_ELBOW_FULL_IDX])
    new_fp[LEFT_ELBOW_FULL_IDX] = (delta * R_bind).as_rotvec()
    G_new = compute_G(model, bind["betas"], new_fp)

    # bind pose 下的世界座標（度量自我接觸用）
    v_bind_world = chupa_v.copy()
    v_bind_raw = (v_bind_world - bind["transl"].reshape(1, 3)) / bind["scale"]

    # 自我接觸區域中心：左手腕（joint 20）的世界座標
    # G_bind 為模型座標系，需套用 scale + translation 轉世界座標
    LEFT_WRIST_IDX = 20
    wrist_model = G_bind[LEFT_WRIST_IDX][:3, 3]
    region_center = wrist_model * bind["scale"] + bind["transl"]
    print(f"[INFO] 自我接觸區域中心（左手腕世界座標）: {region_center}")

    methods = {
        "M1_nearest": lambda: transfer_M1_nearest(chupa_v, source_v, lbs_w),
        "M2_knn_gaussian": lambda: transfer_M2_knn_gaussian(chupa_v, source_v, lbs_w),
        "M3_barycentric": lambda: transfer_M3_barycentric(chupa_v, source_mesh, lbs_w),
        "M4_bodypart": lambda: transfer_M4_bodypart(chupa_v, source_v, lbs_w, source_part),
    }

    results = {}
    for name, fn in methods.items():
        print(f"\n[INFO] === {name} ===")
        weights, _ = fn()

        m1 = metric_adjacency_conflict(weights, faces)
        m2 = metric_selfcontact_leakage(weights, v_bind_world, G_bind,
                                        bind["scale"], bind["transl"])
        v_driven_raw = reskin(v_bind_raw, weights, G_bind, G_new)
        v_driven = v_driven_raw * bind["scale"] + bind["transl"].reshape(1, 3)
        m3 = metric_driven_roughness(v_driven, faces)
        local = metric_local_tearing(v_driven, v_bind_world, faces, region_center)

        results[name] = {"adjacency_conflict": m1,
                         "selfcontact_leakage": m2,
                         "driven_roughness": m3,
                         "local_tearing": local}
        print(f"  [全域] 相鄰衝突率     : {m1:.4f}")
        print(f"  [全域] 自我接觸洩漏率 : {m2:.4f}")
        print(f"  [全域] 驅動後粗糙度   : {m3:.6f}")
        print(f"  [局部] 位移不連續 mean/p95/max: "
              f"{local['disp_discontinuity_mean']:.3f} / "
              f"{local['disp_discontinuity_p95']:.3f} / "
              f"{local['disp_discontinuity_max']:.3f}"
              f"  (區域邊數={local['region_edges']})")
              
        # 輸出驅動後網格供肉眼比對
        trimesh.Trimesh(vertices=v_driven, faces=faces, process=False).export(
            OUT_DIR / f"driven_{name}.obj")
        np.save(OUT_DIR / f"weights_{name}.npy", weights)

    # 匯總表
    print("\n\n========== 匯總（所有指標皆越低越好）==========")
    print("全域指標（整片網格平均）：")
    print(f"{'方法':<18}{'相鄰衝突率':>12}{'自我接觸洩漏':>14}{'驅動後粗糙度':>14}")
    for name, r in results.items():
        print(f"{name:<18}{r['adjacency_conflict']:>12.4f}"
              f"{r['selfcontact_leakage']:>14.4f}{r['driven_roughness']:>14.6f}")

    print("\n局部指標（僅自我接觸區域，位移不連續性，越低越好）：")
    print(f"{'方法':<18}{'不連續mean':>12}{'不連續p95':>12}{'不連續max':>12}")
    for name, r in results.items():
        lt = r["local_tearing"]
        print(f"{name:<18}{lt['disp_discontinuity_mean']:>12.3f}"
              f"{lt['disp_discontinuity_p95']:>12.3f}{lt['disp_discontinuity_max']:>12.3f}")

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] 指標與驅動網格已存至 {OUT_DIR}/")
    print("       請將 driven_M1~M4.obj 匯入 Blender，聚焦左手臂/口袋區域肉眼比對")


if __name__ == "__main__":
    main()