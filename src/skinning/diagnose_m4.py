"""
diagnose_m4.py
診斷 M4 body-part 約束為何未生效。
關鍵：直接沿用 compare_skinning_methods.py 的 PART_OF_JOINT 與函式，
避免因兩處定義不同而誤判。
"""

import collections
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import smplx

# 直接從主腳本 import，確保用的是同一份定義
from compare_skinning_methods import (
    PART_OF_JOINT,
    get_source_part_labels,
    SOURCE_OBJ,
    CHUPA_MESH,
    SMPLX_MODEL_DIR,
)

# ---- 載入資料 ----
source = trimesh.load(SOURCE_OBJ, force="mesh", process=False)
chupa = trimesh.load(CHUPA_MESH, force="mesh", process=False)
source_v = np.asarray(source.vertices, np.float32)
chupa_v = np.asarray(chupa.vertices, np.float32)

model = smplx.create(SMPLX_MODEL_DIR, model_type="smplx", gender="neutral",
                     use_pca=True, num_pca_comps=6, num_betas=10,
                     num_expression_coeffs=10, batch_size=1)
lbs_w = model.lbs_weights.detach().numpy()

# ---- 用主腳本的函式產生 source_part ----
source_part = get_source_part_labels(source_v, lbs_w)
print("source_part 分布:", collections.Counter(source_part))
print("PART_OF_JOINT 分組檢查（各部位含哪些關節）:")
part_to_joints = collections.defaultdict(list)
for j, p in PART_OF_JOINT.items():
    part_to_joints[p].append(j)
for p, js in part_to_joints.items():
    print(f"  {p}: {js}")

# ---- 全域最近點 vs k 近鄰多數決 ----
tree = cKDTree(source_v)

_, idx1 = tree.query(chupa_v, k=1)
part_nearest = source_part[idx1]

k_vote = 12
_, idxk = tree.query(chupa_v, k=k_vote)
pn = sorted(set(source_part))
p2i = {p: i for i, p in enumerate(pn)}
si = np.array([p2i[p] for p in source_part])
part_vote = np.array([
    pn[np.bincount(si[row], minlength=len(pn)).argmax()]
    for row in idxk
])

disagree_mask = (part_nearest != part_vote)
n_disagree = disagree_mask.sum()
print(f"\n部位判定不一致（多數決 vs 全域最近點）: {n_disagree} / {len(chupa_v)} "
      f"({100*n_disagree/len(chupa_v):.2f}%)")

# ---- 若有不一致，看它們是哪些部位之間的混淆，以及空間位置 ----
if n_disagree > 0:
    print("\n不一致頂點的『最近點部位 -> 多數決部位』轉移統計:")
    trans = collections.Counter(
        zip(part_nearest[disagree_mask], part_vote[disagree_mask])
    )
    for (a, b), c in trans.most_common():
        print(f"  {a} -> {b}: {c}")

    # 這些不一致頂點的空間位置（世界座標 y 高度、離身體中心距離）
    dis_v = chupa_v[disagree_mask]
    print(f"\n不一致頂點的空間範圍:")
    print(f"  bbox min: {dis_v.min(0)}")
    print(f"  bbox max: {dis_v.max(0)}")
