r"""
drive_avatar.py

把 uv_texture_bake.py 產出的「帶貼圖的著衣網格」，接上 SMPL-X 蒙皮權重與
ROMP 預測的動作，匯出成單一檔案的 rigged glTF (.glb)。

為什麼是 glTF 而不是每影格一個 obj：
  貼圖與 UV 座標綁在頂點上，跟頂點位置無關，所以整段動畫共用一張貼圖。
  glb 裡只存「一份網格 + 一份權重 + 每影格的關節旋轉」，播放時由引擎做 LBS。
  相較之下每影格輸出 obj 會把 77k 個頂點乘上影格數，檔案大又無法即時播放。

依賴（裝在 chupa 環境，兩個 stage 都需要 smplx）：
  pip install pygltflib          # 純 Python，py3.8 可用

────────────────────────────────────────────────────────────────
跑法
────────────────────────────────────────────────────────────────

  conda activate chupa
  cd ~/projects/chupa

  # 1) 蒙皮權重：把 SMPL-X 的 LBS 權重轉移到 chupa 網格上
  python drive_avatar.py --stage weights \
      --work tex_work \
      --smplx-obj debug_source_smplx_rp_ben_posed_001_v2.obj \
      --bind-pkl data/renderpeople/smplx/rp_ben_posed_001.pkl

  # 2) 匯出 rigged glb（先不給動作，只出 T-pose 綁定，用來驗證骨架對不對）
  python drive_avatar.py --stage export --work tex_work \
      --texture uv_texture_output/chupa_textured_texture.png \
      --out avatar.glb

  # 3) 接上 ROMP 動作（--motion 可以指向整個影格資料夾）
  python drive_avatar.py --stage export --work tex_work \
      --texture uv_texture_output/chupa_textured_texture.png \
      --motion ~/projects/romp_test/img4640_output \
      --fps 30 --out avatar_animated.glb

  把 .glb 拖進 Blender（File → Import → glTF 2.0）或任何 glTF 檢視器
  就能直接播放。Blender 匯入後按空白鍵播放時間軸。

────────────────────────────────────────────────────────────────
重要前提
────────────────────────────────────────────────────────────────
  * --work 要指向 uv_texture_bake.py 用過的同一個資料夾。網格必須是
    geometry.npz 裡那份「xatlas 重新索引後」的頂點，因為貼圖的 UV 是對應
    那份索引的。拿原始 mesh_refine.obj 會對不上。

  * --bind-pkl 是 chupa 生成這個網格時用的那個 SMPL-X 參數檔，
    決定了 bind pose。用錯檔案會導致動起來整個扭曲。

  * ROMP 輸出的是 SMPL（24 關節），SMPL-X body 的前 22 個關節語意相同，
    直接對應；SMPL 的 22/23（手掌根）在 SMPL-X 沒有對應，會被忽略。
    手指與表情維持 bind pose 不動。

  * global_orient 會套 R_CW = diag(1,-1,-1) 校正（ROMP 相機空間 Y 朝下，
    SMPL-X 世界座標 Y 朝上）。少了這步網格會上下顛倒。body_pose 是相對
    父關節的局部旋轉，不受世界座標系影響，不校正。

  * 預設丟棄 ROMP 的位移。角色在畫面裡的位置由綁定資產的 body_scale 與
    global_body_translation 固定，只有姿態改變 —— 虛擬試衣要看的是衣服，
    不需要人物走動。要位移請加 --use-translation（絕對定位沒有已驗證的
    換算關係，只做得到相對第一影格）。

────────────────────────────────────────────────────────────────
可能需要調整的地方
────────────────────────────────────────────────────────────────
  * MAX_INFLUENCES：glTF 標準每個頂點最多 4 個骨骼影響。SMPL-X 權重通常
    集中在少數幾個關節，取前 4 大再正規化的損失很小。如果關節處出現
    明顯撕裂，那通常是權重轉移的問題，不是這裡截斷造成的。

  * 網格上下顛倒 → 檢查 R_CW 校正有沒有被 --no-cam-correction 關掉。

  * 除錯順序：先 --stage export 不給 --motion 確認靜態綁定正確，
    再加動作。這樣能分辨問題出在綁定還是動作。
"""

import argparse
import json
import pickle
import struct
from pathlib import Path

import numpy as np


SMPLX_MODEL_DIR = "src/smpl_related/models"
NUM_JOINTS = 55
NUM_BODY_JOINTS = 21
MAX_INFLUENCES = 4          # glTF 標準上限
N_SMPL_BODY = 22            # SMPL 前 22 個關節與 SMPL-X body 語意相同


# ============================================================
# 共用：SMPL-X 前向運動學（純 numpy，不依賴 torch）
# ============================================================

def rodrigues(rotvecs):
    """axis-angle (N,3) → rotation matrix (N,3,3)。"""
    rotvecs = np.asarray(rotvecs, np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    axis = rotvecs / safe
    c, s = np.cos(theta)[:, 0], np.sin(theta)[:, 0]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    N = len(rotvecs)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c + x * x * (1 - c)
    R[:, 0, 1] = x * y * (1 - c) - z * s
    R[:, 0, 2] = x * z * (1 - c) + y * s
    R[:, 1, 0] = y * x * (1 - c) + z * s
    R[:, 1, 1] = c + y * y * (1 - c)
    R[:, 1, 2] = y * z * (1 - c) - x * s
    R[:, 2, 0] = z * x * (1 - c) - y * s
    R[:, 2, 1] = z * y * (1 - c) + x * s
    R[:, 2, 2] = c + z * z * (1 - c)
    # theta 趨近 0 時退回單位矩陣，避免除零產生 NaN
    R[theta[:, 0] < 1e-8] = np.eye(3)
    return R


def rotvec_to_quat(rotvecs):
    """axis-angle (N,3) → quaternion (N,4)，glTF 慣例順序 (x,y,z,w)。"""
    rotvecs = np.asarray(rotvecs, np.float64).reshape(-1, 3)
    theta = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    axis = rotvecs / safe
    half = theta / 2.0
    q = np.concatenate([axis * np.sin(half), np.cos(half)], axis=1)
    q[theta[:, 0] < 1e-8] = np.array([0.0, 0.0, 0.0, 1.0])
    return q


def global_transforms(J_rest, full_pose, parents):
    """回傳每個關節的『真實全域變換』G_j (J,4,4)。

    注意這裡刻意不用 smplx.lbs.batch_rigid_transform 的回傳值：那個函式回傳的是
    A_j = G_j @ inv(平移到 J_rest_j)，適合直接拿來做 LBS，但 glTF 需要的
    inverseBindMatrix 是 inv(G_j)，兩者差一個平移。自己算比較不會弄錯。
    """
    R = rodrigues(full_pose.reshape(-1, 3))
    G = np.zeros((len(J_rest), 4, 4))
    G[0] = np.eye(4)
    G[0][:3, :3] = R[0]
    G[0][:3, 3] = J_rest[0]
    for j in range(1, len(J_rest)):
        local = np.eye(4)
        local[:3, :3] = R[j]
        local[:3, 3] = J_rest[j] - J_rest[parents[j]]
        G[j] = G[parents[j]] @ local
    return G


def make_full_pose(global_orient, body_pose_63):
    fp = np.zeros((NUM_JOINTS, 3), np.float64)
    fp[0] = np.asarray(global_orient).reshape(3)
    fp[1:1 + NUM_BODY_JOINTS] = np.asarray(body_pose_63).reshape(NUM_BODY_JOINTS, 3)
    return fp


def load_bind(pkl_path):
    """讀 chupa 生成這個網格時用的 SMPL-X 參數（沿用 compare_skinning_methods.py）。"""
    with open(pkl_path, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    go = np.array(p["global_orient"], np.float64).reshape(3)
    betas = np.array(p["betas"], np.float64).reshape(-1)
    bp = np.array(p["body_pose"], np.float64).reshape(-1, 3)
    bp63 = bp[1:22].reshape(63) if len(bp) >= 22 else bp.reshape(-1)[:63]
    scale = float(np.asarray(p["body_scale"]).reshape(-1)[0])
    transl = np.asarray(p["global_body_translation"], np.float64).reshape(3)
    return {"betas": betas, "full_pose": make_full_pose(go, bp63),
            "scale": scale, "transl": transl}


def build_smplx(model_dir, betas):
    """建立 SMPL-X 模型，回傳 (rest 關節位置, parents, LBS 權重, 模板頂點)。"""
    import torch
    import smplx as smplx_lib
    from smplx.lbs import vertices2joints, blend_shapes

    m = smplx_lib.create(model_dir, model_type="smplx", gender="neutral",
                         use_pca=True, num_pca_comps=6, num_betas=10,
                         num_expression_coeffs=10, batch_size=1)
    m.eval()
    betas_t = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    v_shaped = m.v_template.unsqueeze(0) + blend_shapes(
        betas_t, m.shapedirs[:, :, :betas_t.shape[-1]])
    J_rest = vertices2joints(m.J_regressor, v_shaped).squeeze(0).detach().numpy().astype(np.float64)
    parents = m.parents.detach().cpu().numpy()
    lbs_w = m.lbs_weights.detach().cpu().numpy()
    return J_rest, parents, lbs_w, v_shaped.squeeze(0).detach().numpy()


# ============================================================
# Stage 1: weights —— 把 SMPL-X 的 LBS 權重轉移到 chupa 網格
# ============================================================

def stage_weights(work_dir, smplx_obj, bind_pkl, model_dir):
    import trimesh
    from scipy.spatial import cKDTree

    work = Path(work_dir)
    g = np.load(work / "geometry.npz")
    verts = g["verts"].astype(np.float64)
    print(f"[INFO] chupa 網格（xatlas 重新索引後）: {len(verts)} 頂點")

    bind = load_bind(bind_pkl)
    J_rest, parents, lbs_w, _ = build_smplx(model_dir, bind["betas"])

    src = trimesh.load(smplx_obj, force="mesh", process=False)
    src_v = np.asarray(src.vertices, np.float64)
    if len(src_v) != len(lbs_w):
        raise ValueError(
            f"--smplx-obj 有 {len(src_v)} 頂點，SMPL-X 模型是 {len(lbs_w)}。\n"
            f"必須是未經 remesh、維持原始拓樸的 SMPL-X obj。")

    # 權重轉移方法：M1（最近頂點，k=1）。
    #
    # 這是實驗比較後選定的方法，理由不是它指標最好，而是它在這個系統的
    # 實際資產上「弱點不會被觸發」：
    #   * M2（k-NN 高斯 k=12）全域三項指標最佳，但在乾淨 A-pose 資產上
    #     驅動後與 M1 平均只差 0.63mm（身高的 0.05%），視覺無法區分，
    #     而耗時是 M1 的 4.6 倍（163ms vs 35ms）。
    #   * M4（body-part 約束）在相鄰衝突與粗糙度上反而是四者最差，
    #     因為部位交界處會產生硬邊界。
    #   * M1 唯一的弱點是自我接觸（手插口袋那類）會誤配，但綁定資產都是
    #     無自我接觸的 A-pose，所以不會發生。
    #
    # 如果之後改用有自我接觸的 bind pose，這裡要換成 M2；那時要注意
    # body-part 約束若用單一最近點決定部位會完全退化成 M1（權重逐位元相同），
    # 必須用 k=12 多數決決定部位，約束才會真的生效。
    print("[INFO] 轉移蒙皮權重（M1 最近頂點）...")
    _, idx = cKDTree(src_v).query(verts, k=1)
    W = lbs_w[idx]

    # glTF 每頂點最多 4 個骨骼影響：取前 4 大再正規化。
    order = np.argsort(-W, axis=1)[:, :MAX_INFLUENCES]
    joints = order.astype(np.uint16)
    weights = np.take_along_axis(W, order, axis=1).astype(np.float32)
    s = weights.sum(axis=1, keepdims=True)
    weights = weights / np.where(s < 1e-8, 1.0, s)

    kept = float(np.mean(np.take_along_axis(W, order, axis=1).sum(axis=1)))
    print(f"[INFO] 取前 {MAX_INFLUENCES} 大權重後保留了平均 {100 * kept:.2f}% 的權重質量")
    if kept < 0.98:
        print("       (低於 98% 代表權重很分散，關節處可能會有輕微變形)")

    np.savez_compressed(work / "skin_weights.npz", joints=joints, weights=weights)
    print(f"[DONE] 權重存到 {work / 'skin_weights.npz'}")


def _joint_part(j):
    if j in [16, 18, 20] or 25 <= j < 40:
        return "left_arm"
    if j in [17, 19, 21] or 40 <= j < 55:
        return "right_arm"
    if j in [1, 4, 7, 10]:
        return "left_leg"
    if j in [2, 5, 8, 11]:
        return "right_leg"
    return "torso"


# ============================================================
# 動作載入（ROMP 輸出）
# ============================================================

POSE_KEYS = ["smpl_thetas", "poses", "pose", "body_pose", "global_orient_and_pose"]
TRANS_KEYS = ["cam_trans", "trans", "translation", "global_body_translation"]


def _unwrap(raw):
    """simple_romp 把每影格的結果包在 'results' 裡（0 維 object 陣列），
    先剝掉這層外殼拿到真正的 dict。
    """
    if isinstance(raw, dict) and "results" in raw:
        r = raw["results"]
        if isinstance(r, np.ndarray):
            r = r.item() if r.ndim == 0 else r[0]
        if isinstance(r, dict):
            return r
    return raw


def _load_one(path):
    if path.suffix == ".npz":
        raw = dict(np.load(path, allow_pickle=True))
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            raw = pickle.load(f, encoding="latin1")
    else:
        raise ValueError(f"不認得的格式: {path.suffix}（支援 .npz / .pkl）")
    return _unwrap(raw)


def _pick(d, keys):
    for k in keys:
        if k in d:
            return k, np.asarray(d[k], np.float64)
    return None, None


def _frame_keyed(d):
    """判斷這個 dict 是不是「以影格檔名為鍵」的彙總結構。

    simple_romp 跑影片時除了每影格一個 npz，還會存一個 video_results.npz，
    它的鍵是 '00000000.png' 這種影格圖檔名、值才是該影格的結果 dict。
    直接讀這個彙總檔比讀幾百個單檔快得多。
    """
    if not isinstance(d, dict) or len(d) < 2:
        return False
    ks = list(d.keys())
    if not all(isinstance(k, str) for k in ks):
        return False
    if not any(k.lower().endswith((".png", ".jpg", ".jpeg")) for k in ks[:3]):
        return False
    for k in ks[:3]:
        v = d[k]
        if isinstance(v, dict):
            continue
        if isinstance(v, np.ndarray) and v.dtype == object:
            continue
        return False
    return True


def _extract_frame(d, person):
    """從單一影格的結果 dict 抽出 (pose, trans)。沒偵測到人時回傳 (None, None)。"""
    if isinstance(d, np.ndarray):
        d = d.item() if d.ndim == 0 else d[0]
    d = _unwrap(d)

    pk, pv = _pick(d, POSE_KEYS)
    if pv is None:
        raise KeyError(f"找不到姿態資料。實際有的鍵：{list(d.keys())}")
    pv = pv.reshape(-1, pv.shape[-1]) if pv.ndim > 1 else pv.reshape(1, -1)
    if len(pv) == 0:
        return None, None, pk, None
    pose = pv[min(person, len(pv) - 1)]

    tk, tv = _pick(d, TRANS_KEYS)
    tr = None
    if tv is not None:
        tv = tv.reshape(-1, tv.shape[-1]) if tv.ndim > 1 else tv.reshape(1, -1)
        if len(tv):
            tr = tv[min(person, len(tv) - 1)][:3]
    return pose, tr, pk, tk


def _assemble(frames, person, label):
    """把一連串影格結果組成 (poses, trans)。"""
    poses, trans, skipped = [], [], 0
    pose_key = trans_key = None
    for item in frames:
        pose, tr, pk, tk = _extract_frame(item, person)
        pose_key = pose_key or pk
        trans_key = trans_key or tk
        if pose is None:
            skipped += 1
            continue
        poses.append(pose)
        if tr is not None:
            trans.append(tr)
    if not poses:
        raise ValueError("所有影格都沒有偵測到人，無法產生動作")
    if skipped:
        print(f"[WARN] {skipped} 個影格沒偵測到人，已跳過"
              f"（動作會在該處接起來，可能有跳動）")
    poses = np.stack(poses)
    trans = np.stack(trans) if len(trans) == len(poses) else None
    print(f"[INFO] {label}：姿態鍵 '{pose_key}'，shape={poses.shape}")
    print(f"[INFO] 位移鍵 '{trans_key}'" if trans is not None
          else "[WARN] 沒有位移資料，角色會原地動作")
    return poses, trans


def load_motion(path, person=0):
    """讀 ROMP 輸出的動作序列。

    支援兩種形式：
      * 資料夾 —— 每影格一個 .npz（simple_romp 處理影片的預設輸出），
        依檔名排序後串成序列。
      * 單一檔案 —— 整段動作在一個檔裡，或只有單一影格（靜態姿勢）。

    ROMP 會偵測多個人，用 person 參數挑一個（預設第 0 個）。

    回傳 (poses (F,72), trans (F,3) 或 None)。
    """
    path = Path(path)

    # ── 資料夾：優先用彙總檔，沒有才讀每影格單檔 ──
    if path.is_dir():
        summary = path / "video_results.npz"
        if summary.exists():
            print(f"[INFO] 找到彙總檔 {summary.name}，直接用它（比讀幾百個單檔快）")
            return load_motion(summary, person)

        files = sorted(f for f in path.glob("*.npz")
                       if not f.name.startswith("video_results"))
        files = files or sorted(path.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"{path} 裡沒有可用的 .npz / .pkl 檔")
        print(f"[INFO] 從資料夾讀到 {len(files)} 個影格檔 "
              f"({files[0].name} … {files[-1].name})")
        return _assemble([_load_one(f) for f in files], person, "逐檔讀取")

    # ── 單一檔案 ──
    d = _load_one(path)

    # 彙總檔：鍵是影格圖檔名，依檔名排序後就是時間順序
    if _frame_keyed(d):
        keys = sorted(d.keys())
        print(f"[INFO] 彙總檔含 {len(keys)} 個影格 ({keys[0]} … {keys[-1]})")
        return _assemble([d[k] for k in keys], person, "彙總檔")

    # 攤平的整段動作，或單一姿勢
    pk, poses = _pick(d, POSE_KEYS)
    if poses is None:
        raise KeyError(
            f"在 {path} 裡找不到姿態資料。實際有的鍵：{list(d.keys())}\n"
            f"請把這個清單告訴我，我幫你對應。")
    poses = poses.reshape(-1, poses.shape[-1]) if poses.ndim > 1 else poses.reshape(1, -1)
    print(f"[INFO] 姿態鍵 '{pk}'，shape={poses.shape}")
    if len(poses) == 1:
        print("[WARN] 只有單一姿勢，輸出會是靜態的。"
              "要動畫請把 --motion 指向整個影格資料夾。")

    tk, trans = _pick(d, TRANS_KEYS)
    if trans is not None:
        trans = trans.reshape(-1, trans.shape[-1])[:len(poses), :3]
        print(f"[INFO] 位移鍵 '{tk}'")
    else:
        print("[WARN] 沒有位移資料，角色會原地動作")
    return poses, trans


# ROMP 相機空間 → SMPL-X 世界座標的校正矩陣。
# ROMP 的相機空間 Y 軸朝下，SMPL-X 世界座標 Y 軸朝上，差一個繞 X 軸 180 度。
# 只作用在 global_orient（根旋轉）—— body_pose 是相對父關節的局部旋轉，
# 不受世界座標系選擇影響，套上去反而會壞掉。
# 少了這步網格會上下顛倒。
R_CW = np.diag([1.0, -1.0, -1.0])


def motion_to_full_pose(poses, bind_full_pose, apply_cam_correction=True):
    """把 ROMP 的 SMPL 姿態轉成 SMPL-X 的 55 關節 full pose。

    SMPL theta 是 (24,3)：第 0 個是 global_orient，1~21 對應 SMPL-X 的身體關節，
    22/23（手掌根）在 SMPL-X 沒有對應關節，捨棄。
    手指與表情關節維持 bind pose 不動 —— ROMP 本來就不預測這些。
    """
    from scipy.spatial.transform import Rotation

    F = len(poses)
    out = np.tile(bind_full_pose.reshape(1, NUM_JOINTS, 3), (F, 1, 1))
    n = poses.shape[1] // 3
    take = min(n, N_SMPL_BODY, NUM_JOINTS)
    out[:, :take] = poses[:, :take * 3].reshape(F, take, 3)

    if apply_cam_correction:
        mats = Rotation.from_rotvec(out[:, 0]).as_matrix()
        out[:, 0] = Rotation.from_matrix(R_CW @ mats).as_rotvec()
        print("[INFO] 已對 global_orient 套用 ROMP 相機空間 → 世界座標校正")

    print(f"[INFO] 動作共 {F} 影格，套用前 {take} 個關節（其餘維持 bind pose）")
    return out


# ============================================================
# Stage 2: export —— 組出 rigged glb
# ============================================================

class GLBBuilder:
    """把資料塞進單一 binary chunk，並產生對應的 bufferView / accessor。

    glTF 的 accessor 要求 byteOffset 對齊到 component 大小，所以每次寫入
    都補到 4 bytes 邊界。這裡把細節包起來，呼叫端只要給 numpy 陣列。
    """

    COMP = {np.dtype(np.float32): 5126, np.dtype(np.uint16): 5123,
            np.dtype(np.uint32): 5125, np.dtype(np.uint8): 5121}
    TYPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}

    def __init__(self):
        self.blob = bytearray()
        self.views = []
        self.accessors = []

    def _pad(self):
        while len(self.blob) % 4:
            self.blob.append(0)

    def add_view(self, data_bytes, target=None):
        self._pad()
        off = len(self.blob)
        self.blob.extend(data_bytes)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data_bytes)}
        if target is not None:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def add_accessor(self, arr, target=None, minmax=False):
        arr = np.ascontiguousarray(arr)
        ncomp = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
        vi = self.add_view(arr.tobytes(), target)
        acc = {"bufferView": vi, "componentType": self.COMP[arr.dtype],
               "count": int(arr.shape[0]), "type": self.TYPE[ncomp]}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def stage_export(work_dir, texture_path, motion_path, bind_pkl, model_dir, fps, out_path,
                 person=0, use_translation=False, no_cam_correction=False,
                 x_offset=0.0, align_feet=False):
    work = Path(work_dir)
    g = np.load(work / "geometry.npz")
    verts_world = g["verts"].astype(np.float64)
    faces = g["faces"].astype(np.uint32)

    # UV 的 v 軸慣例：OBJ 原點在左下（v 往上增加），glTF 原點在左上（v 往下增加）。
    # 貼圖是照 OBJ 慣例烘的，直接塞進 glTF 等於整張圖上下翻轉。
    # 因為 xatlas 的 UV 島是散布的，翻轉後每個島會落到別的島的位置，
    # 症狀是「顏色種類正確但位置全部錯亂」的拼貼亂碼。
    uv = g["uv"].astype(np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]

    wpath = work / "skin_weights.npz"
    if not wpath.exists():
        raise FileNotFoundError(f"找不到 {wpath}，請先跑 --stage weights")
    sw = np.load(wpath)
    skin_joints, skin_weights = sw["joints"], sw["weights"]

    bind = load_bind(bind_pkl)
    J_rest, parents, _, _ = build_smplx(model_dir, bind["betas"])

    # 網格從 chupa 的世界座標換回 SMPL-X 標準空間，
    # 這樣才跟 J_rest / 前向運動學在同一個座標系裡。
    verts = (verts_world - bind["transl"].reshape(1, 3)) / bind["scale"]

    G_bind = global_transforms(J_rest, bind["full_pose"], parents)
    ibm = np.linalg.inv(G_bind).astype(np.float32)
    # glTF 的矩陣是 column-major，numpy 是 row-major，要轉置
    ibm_gltf = np.transpose(ibm, (0, 2, 1)).copy()

    b = GLBBuilder()
    a_pos = b.add_accessor(verts.astype(np.float32), target=34962, minmax=True)
    a_uv = b.add_accessor(uv, target=34962)
    a_joints = b.add_accessor(skin_joints.astype(np.uint16), target=34962)
    a_weights = b.add_accessor(skin_weights.astype(np.float32), target=34962)
    a_idx = b.add_accessor(faces.reshape(-1).astype(np.uint32), target=34963)
    a_ibm = b.add_accessor(ibm_gltf.reshape(len(J_rest), 16))

    # ── 節點階層：先是骨架，最後一個是掛網格的節點 ──
    #
    # x_offset 加在根關節而不是網格節點上：依 glTF 規範，蒙皮網格節點自身的
    # 變換會被忽略（一切由 skin 的關節決定），所以位移必須從骨架根部下手。
    # 把腳底對齊到 y=0。
    #
    # 所有 subject 共用模板的 global_body_translation，骨盆位置固定，
    # 於是不同體型的腳會落在不同高度。並排比較時基準線不齊，
    # 人眼幾乎無法判斷身高差異 —— 17% 的差距會被看成「都差不多」。
    y_offset = -float(verts[:, 1].min()) if align_feet else 0.0
    if align_feet:
        print(f"[INFO] 腳底對齊 y=0（位移 {y_offset:+.4f}）")

    nodes = []
    bind_quat = rotvec_to_quat(bind["full_pose"])
    for j in range(len(J_rest)):
        t = J_rest[j] if parents[j] < 0 else J_rest[j] - J_rest[parents[j]]
        if parents[j] < 0 and (x_offset or y_offset):
            t = t + np.array([x_offset, y_offset, 0.0])
        node = {"name": f"joint_{j}",
                "translation": [float(x) for x in t],
                "rotation": [float(x) for x in bind_quat[j]]}
        children = [int(c) for c in np.where(parents == j)[0]]
        if children:
            node["children"] = children
        nodes.append(node)

    mesh_node = len(nodes)
    nodes.append({"name": "avatar", "mesh": 0, "skin": 0})

    # ── 動作 ──
    animations = []
    if motion_path:
        poses, trans = load_motion(motion_path, person=person)
        # 預設丟棄位移。角色在畫面裡的位置由綁定資產的 body_scale 與
        # global_body_translation 決定，固定不變，只有姿態改變。
        # 這對虛擬試衣是正確的行為 —— 使用者要看的是衣服，不需要人物在
        # 畫面裡跑來跑去。ROMP 的 cam_trans 是相機空間位移，跟這個世界
        # 座標系沒有已驗證的換算關係，硬接上去只會讓人飄走。
        if not use_translation:
            trans = None
        full = motion_to_full_pose(poses, bind["full_pose"],
                                   apply_cam_correction=not no_cam_correction)
        F = len(full)
        times = (np.arange(F, dtype=np.float32) / float(fps))
        a_time = b.add_accessor(times, minmax=True)

        channels, samplers = [], []
        n_anim_joints = min(N_SMPL_BODY, NUM_JOINTS)
        for j in range(n_anim_joints):
            q = rotvec_to_quat(full[:, j]).astype(np.float32)
            a_rot = b.add_accessor(q)
            samplers.append({"input": a_time, "output": a_rot, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": j, "path": "rotation"}})

        if trans is not None:
            # 只有明確加了 --use-translation 才會走到這裡。用相對第一影格的位移，
            # 至少保證起始位置正確；絕對定位沒有已驗證的換算關係，會飄。
            root_t = J_rest[0].reshape(1, 3) + (trans - trans[0:1]) / bind["scale"]
            if x_offset or y_offset:
                root_t = root_t + np.array([[x_offset, y_offset, 0.0]])
            a_tr = b.add_accessor(root_t.astype(np.float32))
            samplers.append({"input": a_time, "output": a_tr, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": 0, "path": "translation"}})

        animations.append({"name": "romp_motion", "channels": channels, "samplers": samplers})
        print(f"[INFO] 動畫: {F} 影格 @ {fps}fps = {F / fps:.1f} 秒")
    else:
        print("[INFO] 未指定 --motion，只輸出靜態綁定（用來檢查骨架是否正確）")

    # ── 貼圖 ──
    tex_bytes = Path(texture_path).read_bytes()
    img_view = b.add_view(tex_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "drive_avatar.py"},
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node]}],
        "nodes": nodes,
        "meshes": [{"name": "avatar_mesh", "primitives": [{
            "attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv,
                           "JOINTS_0": a_joints, "WEIGHTS_0": a_weights},
            "indices": a_idx, "material": 0}]}],
        "skins": [{"inverseBindMatrices": a_ibm,
                   "joints": list(range(len(J_rest))), "skeleton": 0}],
        "materials": [{"name": "avatar_mat", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0.0, "roughnessFactor": 0.9}}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": img_view, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}],
        "bufferViews": b.views,
        "accessors": b.accessors,
        "buffers": [{"byteLength": len(b.blob)}],
    }
    if animations:
        gltf["animations"] = animations

    _write_glb(gltf, bytes(b.blob), out_path)
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[DONE] {out_path} ({size_mb:.1f} MB)")
    print(f"       Blender: File → Import → glTF 2.0，匯入後按空白鍵播放")


def _write_glb(gltf_dict, bin_blob, out_path):
    """把 JSON 與 binary 打包成單一 .glb（比 .gltf+.bin+.png 三個檔好帶）。"""
    js = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    blob = bin_blob + b"\x00" * ((4 - len(bin_blob) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(blob)


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["weights", "export"])
    p.add_argument("--work", default="tex_work", help="uv_texture_bake.py 用的同一個資料夾")
    p.add_argument("--smplx-obj", default="debug_source_smplx_rp_ben_posed_001_v2.obj")
    p.add_argument("--bind-pkl", default="data/renderpeople/smplx/rp_ben_posed_001.pkl")
    p.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR)
    p.add_argument("--texture", default="uv_texture_output/chupa_textured_texture.png")
    p.add_argument("--motion", default=None,
                   help="ROMP 輸出。可以是整個影格資料夾（每影格一個 npz），"
                        "或單一檔案。不給就只出靜態綁定")
    p.add_argument("--person", type=int, default=0,
                   help="影片裡有多人時要用哪一個（預設第 0 個）")
    p.add_argument("--use-translation", action="store_true",
                   help="套用 ROMP 的位移讓角色在畫面裡移動。預設關閉 —— "
                        "位置由綁定資產固定，虛擬試衣不需要人物走動")
    p.add_argument("--no-cam-correction", action="store_true",
                   help="關閉 ROMP 相機空間→世界座標的校正。"
                        "只有在動作來源不是 ROMP 時才需要")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--x-offset", type=float, default=0.0,
                   help="沿 X 軸位移。多個 glb 各給不同值，匯入 Blender 時"
                        "就會自動排成一列不重疊，適合做並排對照")
    p.add_argument("--align-feet", action="store_true",
                   help="把腳底對齊到 y=0。體型對照必加 —— 基準線不齊時"
                        "人眼看不出身高差異")
    p.add_argument("--out", default="avatar.glb")
    args = p.parse_args()

    if args.stage == "weights":
        stage_weights(args.work, args.smplx_obj, args.bind_pkl, args.smplx_model_dir)
    else:
        stage_export(args.work, args.texture, args.motion, args.bind_pkl,
                     args.smplx_model_dir, args.fps, args.out,
                     person=args.person, use_translation=args.use_translation,
                     no_cam_correction=args.no_cam_correction,
                     x_offset=args.x_offset, align_feet=args.align_feet)


if __name__ == "__main__":
    main()