r"""
render_overlay.py

把生成的著衣網格「穿」回原始影片裡的人身上，輸出三格對照影片：

    左：原始影片        中：衣服疊回使用者身上        右：3D 著衣模型

只重疊上衣與下著，頭、手、鞋維持原始畫面，所以看起來是使用者本人穿上生成的衣服。

在 chupa 環境執行（需要 nvdiffrast / smplx / trimesh / opencv）。

────────────────────────────────────────────────────────────────
運作方式
────────────────────────────────────────────────────────────────

1. 用 ROMP 每影格的姿態做 LBS，把著衣網格擺成該影格的姿勢。

2. 對位：把我們的 SMPL-X 關節與 ROMP 的 3D 關節做相似變換擬合
   （Umeyama，求縮放 + 旋轉 + 平移），把網格搬到 ROMP 的座標系。
   接著用 ROMP 的 2D 關節（pj2d_org，原圖像素座標）再擬合一次
   2D 的縮放與平移，確保投影後確實落在影片中的人身上。

   為什麼分兩段：ROMP 的相機是弱透視，等同正交投影加縮放，
   所以「3D 對齊 + 2D 擬合」比反推相機內參穩健得多，
   也不必假設 ROMP 各版本的相機參數格式一致。

3. 以 nvdiffrast 光柵化，用衣物遮罩當 alpha 疊回原始畫面。
   遮罩來自 semantic stage 產生的 label_atlas，插值後的邊界值
   直接當作柔邊 alpha，避免鋸齒。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  conda activate chupa
  cd ~/projects/chupa

  # 先看 ROMP 輸出有哪些欄位可用（1 秒，強烈建議先跑）
  python render_overlay.py --inspect --motion ~/projects/romp_test/img4640_output

  # 先渲染前 30 影格確認對位（約 1 分鐘）
  python render_overlay.py \
      --video ~/projects/romp_test/img4640.mp4 \
      --motion ~/projects/romp_test/img4640_output \
      --work tex_work_seed \
      --texture tex_work_seed/final/chupa_textured_texture.png \
      --bind-pkl data/renderpeople/smplx/rp_emily_apose_001.pkl \
      --limit 30 --out /tmp/overlay_test.mp4

  # 確認沒問題後跑完整段
  python render_overlay.py ... --out /tmp/overlay_full.mp4

────────────────────────────────────────────────────────────────
對位不準時怎麼調
────────────────────────────────────────────────────────────────
  衣服整體偏移      → 檢查 --motion 的 pj2d_org 是否為原圖座標
                       （某些版本是正規化的 [-1,1]，用 --inspect 看數值範圍）
  衣服比人大或小    → ALIGN_SCALE 微調
  衣服上下顛倒      → 對位用的關節對應錯了，看 --inspect 的關節數
  衣服穿在別人身上  → 影片裡有多人，用 --person 指定
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# 參數
# ============================================================

SMPLX_MODEL_DIR = "src/smpl_related/models"
NUM_JOINTS = 55
NUM_BODY_JOINTS = 21
N_SMPL_BODY = 22

R_CW = np.diag([1.0, -1.0, -1.0])       # ROMP 相機空間 → SMPL-X 世界座標

# 疊合哪些語意區域。1=上衣 2=下著 3=頭 4=手 5=鞋（見 uv_texture_bake.py）
OVERLAY_REGIONS = [1, 2]

ALIGN_SCALE = 1.0        # 衣服比人大就調小，例如 0.97
PANEL_H = 720            # 每格畫面高度


# ============================================================
# SMPL-X 前向運動學（與 chupa_reskin.py 一致）
# ============================================================

def rodrigues(rv):
    rv = np.asarray(rv, np.float64).reshape(-1, 3)
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    safe = np.where(th < 1e-8, 1.0, th)
    ax = rv / safe
    c, s = np.cos(th)[:, 0], np.sin(th)[:, 0]
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    R = np.zeros((len(rv), 3, 3))
    R[:, 0, 0] = c + x * x * (1 - c); R[:, 0, 1] = x * y * (1 - c) - z * s
    R[:, 0, 2] = x * z * (1 - c) + y * s; R[:, 1, 0] = y * x * (1 - c) + z * s
    R[:, 1, 1] = c + y * y * (1 - c); R[:, 1, 2] = y * z * (1 - c) - x * s
    R[:, 2, 0] = z * x * (1 - c) - y * s; R[:, 2, 1] = z * y * (1 - c) + x * s
    R[:, 2, 2] = c + z * z * (1 - c)
    R[th[:, 0] < 1e-8] = np.eye(3)
    return R


def fk(J_rest, full_pose, parents):
    """回傳每個關節的全域變換 (J,4,4)。"""
    R = rodrigues(full_pose.reshape(-1, 3))
    G = np.zeros((len(J_rest), 4, 4))
    G[0] = np.eye(4); G[0][:3, :3] = R[0]; G[0][:3, 3] = J_rest[0]
    for j in range(1, len(J_rest)):
        L = np.eye(4)
        L[:3, :3] = R[j]
        L[:3, 3] = J_rest[j] - J_rest[parents[j]]
        G[j] = G[parents[j]] @ L
    return G


def make_full_pose(go, bp63):
    fp = np.zeros((NUM_JOINTS, 3), np.float64)
    fp[0] = np.asarray(go).reshape(3)
    fp[1:1 + NUM_BODY_JOINTS] = np.asarray(bp63).reshape(NUM_BODY_JOINTS, 3)
    return fp


def load_bind(pkl_path):
    with open(pkl_path, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    go = np.array(p["global_orient"], np.float64).reshape(3)
    betas = np.array(p["betas"], np.float64).reshape(-1)
    bp = np.array(p["body_pose"], np.float64).reshape(-1, 3)
    bp63 = bp[1:22].reshape(63)          # 丟棄 slot 0
    return {"betas": betas,
            "full_pose": make_full_pose(go, bp63),
            "scale": float(np.asarray(p["body_scale"]).reshape(-1)[0]),
            "transl": np.asarray(p["global_body_translation"], np.float64).reshape(3)}


def build_smplx(model_dir, betas):
    import torch, smplx
    from smplx.lbs import vertices2joints, blend_shapes
    m = smplx.create(model_dir, model_type="smplx", gender="neutral",
                     use_pca=True, num_pca_comps=6, num_betas=10,
                     num_expression_coeffs=10, batch_size=1)
    m.eval()
    bt = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    v_shaped = m.v_template.unsqueeze(0) + blend_shapes(
        bt, m.shapedirs[:, :, :bt.shape[-1]])
    J = vertices2joints(m.J_regressor, v_shaped).squeeze(0).detach().numpy().astype(np.float64)
    return J, m.parents.detach().cpu().numpy()


# ============================================================
# ROMP 輸出
# ============================================================

POSE_KEYS = ["smpl_thetas", "poses", "pose"]
J3D_KEYS = ["joints", "j3d", "smpl_joints", "joints_smpl"]
J2D_KEYS = ["pj2d_org", "pj2d", "joints_2d", "kp2d"]


def _unwrap(d):
    if isinstance(d, dict) and "results" in d:
        r = d["results"]
        if isinstance(r, np.ndarray):
            r = r.item() if r.ndim == 0 else r[0]
        if isinstance(r, dict):
            return r
    return d


def _pick(d, keys):
    for k in keys:
        if k in d:
            return k, np.asarray(d[k], np.float64)
    return None, None


def load_romp_frames(path):
    """回傳 [(pose72, j3d, j2d), ...]，依影格順序。缺的欄位為 None。"""
    path = Path(path)
    if path.is_dir():
        summary = path / "video_results.npz"
        if summary.exists():
            path = summary
        else:
            files = sorted(f for f in path.glob("*.npz")
                           if not f.name.startswith("video_results"))
            return [_extract(_unwrap(dict(np.load(f, allow_pickle=True))))
                    for f in files]

    raw = _unwrap(dict(np.load(path, allow_pickle=True)))
    # 彙總檔：鍵是影格圖檔名
    ks = list(raw.keys())
    if ks and all(isinstance(k, str) for k in ks) and \
       any(k.lower().endswith((".png", ".jpg")) for k in ks[:3]):
        return [_extract(raw[k]) for k in sorted(ks)]
    return [_extract(raw)]


def _extract(d, person=0):
    if isinstance(d, np.ndarray):
        d = d.item() if d.ndim == 0 else d[0]
    d = _unwrap(d)

    def take(keys, dim):
        k, v = _pick(d, keys)
        if v is None:
            return None
        v = v.reshape(-1, *v.shape[1:]) if v.ndim > dim else v[None]
        if len(v) == 0:
            return None
        return v[min(person, len(v) - 1)]

    return (take(POSE_KEYS, 1), take(J3D_KEYS, 2), take(J2D_KEYS, 2))


def inspect(path):
    """把 ROMP 輸出的欄位與數值範圍印出來 —— 對位邏輯依賴這些格式。"""
    path = Path(path)
    f = path / "video_results.npz" if path.is_dir() else path
    if not f.exists():
        cands = sorted(path.glob("*.npz"))
        f = cands[0] if cands else None
    if f is None:
        print(f"找不到 npz: {path}")
        return

    raw = _unwrap(dict(np.load(f, allow_pickle=True)))
    ks = list(raw.keys())
    if ks and any(str(k).lower().endswith((".png", ".jpg")) for k in ks[:3]):
        print(f"彙總檔，{len(ks)} 影格。取第一格檢視：")
        d = raw[sorted(ks)[0]]
        d = d.item() if isinstance(d, np.ndarray) and d.ndim == 0 else d
        d = _unwrap(d)
    else:
        d = raw

    print(f"\n檔案: {f}")
    print(f"欄位: {list(d.keys())}\n")
    for k, v in d.items():
        v = np.asarray(v)
        if v.dtype.kind in "fiu" and v.size:
            print(f"  {k:<16} shape={str(v.shape):<16} "
                  f"範圍 [{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k:<16} shape={v.shape} dtype={v.dtype}")
    print("\n要用到的欄位：")
    for name, keys in [("姿態", POSE_KEYS), ("3D 關節", J3D_KEYS), ("2D 關節", J2D_KEYS)]:
        k, _ = _pick(d, keys)
        print(f"  {name:<8}: {k or '找不到 —— 對位會失敗，請回報這份清單'}")
    print("\n注意：2D 關節若範圍在 [-1,1] 是正規化座標，需乘上影像尺寸；"
          "若範圍是幾百到上千則已是像素座標。")


# ============================================================
# 對位
# ============================================================

def umeyama(src, dst):
    """求相似變換 (scale, R, t) 使 scale*R@src + t ≈ dst。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    C = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(S) @ D) / max(var, 1e-12)
    return scale, R, mu_d - scale * R @ mu_s


def fit_2d(src_xy, dst_xy):
    """求 2D 的等向縮放與平移，使 s*src + t ≈ dst。"""
    mu_s, mu_d = src_xy.mean(0), dst_xy.mean(0)
    s0, d0 = src_xy - mu_s, dst_xy - mu_d
    denom = (s0 ** 2).sum()
    s = float((s0 * d0).sum() / max(denom, 1e-12))
    return s, mu_d - s * mu_s


# ============================================================
# 主流程
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="原始影片檔")
    p.add_argument("--motion", required=True, help="ROMP 輸出資料夾或彙總檔")
    p.add_argument("--work", default="tex_work_seed", help="含 geometry/label/weights 的資料夾")
    p.add_argument("--texture", default=None)
    p.add_argument("--bind-pkl", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    p.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR)
    p.add_argument("--out", default="/tmp/overlay.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0, help="只處理前 N 影格（測試用）")
    p.add_argument("--person", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0, help="疊合不透明度")
    p.add_argument("--regions", type=int, nargs="+", default=OVERLAY_REGIONS,
                   help="要疊合的語意區域。1=上衣 2=下著 3=頭 4=手 5=鞋")
    p.add_argument("--scale", type=float, default=ALIGN_SCALE,
                   help="整體縮放微調。衣服比人大就調小，例如 0.95")
    p.add_argument("--y-shift", type=float, default=0.0,
                   help="垂直微調（像素，正值往下）。衣領蓋到臉就調正值")
    p.add_argument("--x-shift", type=float, default=0.0, help="水平微調（像素）")
    p.add_argument("--debug-mask", action="store_true",
                   help="右格改顯示疊合遮罩（白=會被蓋住），用來確認遮罩範圍")
    p.add_argument("--no-tex-flip-v", dest="tex_flip_v", action="store_false",
                   help="關閉貼圖 V 軸翻轉。顏色種類正確但位置錯亂時切換看看")
    p.set_defaults(tex_flip_v=True)
    p.add_argument("--inspect", action="store_true", help="只檢視 ROMP 輸出格式")
    args = p.parse_args()

    if args.inspect:
        inspect(args.motion)
        return

    if not args.video:
        p.error("需要 --video（或用 --inspect 檢視格式）")

    import torch
    import nvdiffrast.torch as dr

    work = Path(args.work)
    g = np.load(work / "geometry.npz")
    verts_w, faces, uv = (g["verts"].astype(np.float64),
                          g["faces"].astype(np.int32),
                          g["uv"].astype(np.float32))
    sw = np.load(work / "skin_weights.npz")
    sj, swt = sw["joints"].astype(np.int64), sw["weights"].astype(np.float64)

    label = np.load(work / "label_atlas.npy")
    tex_path = args.texture or str(work / "texture.png")
    tex = cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 衣物遮罩：先在 UV 空間標出要疊合的區域，再依每個頂點的 UV 取樣成頂點屬性。
    # 用頂點屬性而非在貼圖空間做，光柵化時插值出來的邊界值可以直接當柔邊 alpha。
    res = label.shape[0]
    tx = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
    ty = np.clip(((1 - uv[:, 1]) * res).astype(int), 0, res - 1)
    # 把標籤填滿 xatlas 的打包間隙後再取樣。
    #
    # 頂點的 UV 換算成整數 texel 時，位於 UV 島邊界的頂點常會落在島與島之間的
    # 間隙上（標籤 0），被判定為「不疊合」，結果是衣服上佈滿細小的破洞，
    # 原始影片會從洞裡透出來。貼圖本身有做邊界外擴所以顏色正常，
    # 但標籤圖是離散值、沒有做外擴，必須在這裡補。
    #
    # 用最近非零標籤填補（而非形態學膨脹），區域交界處才不會偏向某一邊。
    from scipy.ndimage import distance_transform_edt
    if (label == 0).any():
        _, (iy, ix) = distance_transform_edt(label == 0, return_indices=True)
        label_filled = label[iy, ix]
        n_gap = int((label == 0).sum())
        print(f"[INFO] 已填補標籤圖的打包間隙（{n_gap} texel）")
    else:
        label_filled = label
    label = label_filled

    sampled = label[ty, tx]
    vmask = np.isin(sampled, args.regions).astype(np.float32)
    names = {0: "(間隙)", 1: "上衣", 2: "下著", 3: "頭", 4: "手", 5: "鞋"}
    print(f"[INFO] 頂點的語意區域分布：")
    for rid in sorted(set(sampled.tolist())):
        n = int((sampled == rid).sum())
        mark = " ← 疊合" if rid in args.regions else ""
        print(f"       {names.get(rid, rid):<8} {n:>7} ({100*n/len(sampled):5.1f}%){mark}")
    print(f"[INFO] 疊合區域 {args.regions}，佔 {100 * vmask.mean():.1f}% 頂點")

    bind = load_bind(args.bind_pkl)
    J_rest, parents = build_smplx(args.smplx_model_dir, bind["betas"])
    G_bind = fk(J_rest, bind["full_pose"], parents)
    G_bind_inv = np.linalg.inv(G_bind)
    v_raw = (verts_w - bind["transl"].reshape(1, 3)) / bind["scale"]
    v_h = np.concatenate([v_raw, np.ones((len(v_raw), 1))], axis=1)

    frames = load_romp_frames(args.motion)
    print(f"[INFO] ROMP 影格數: {len(frames)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 影片 {W}x{H}, {n_video} 影格")
    if abs(n_video - len(frames)) > 5:
        print(f"[WARN] 影片影格數與 ROMP 影格數差距較大，"
              f"對位會逐格套用，數量不符的部分會被截斷")

    ctx = dr.RasterizeCudaContext()
    faces_t = torch.tensor(faces, dtype=torch.int32, device="cuda")

    # 貼圖採樣用的 UV 要翻轉 V 軸。
    #
    # 烘貼圖時 texel 的列是用 (1-v)*RES 算的（v=1 對應第 0 列），
    # 但 nvdiffrast 的 dr.texture() 把 v 直接當成列的比例（v=0 對應第 0 列），
    # 兩者相反。不翻轉的話每個 UV 島會取到鏡射位置的內容，
    # 症狀是顏色種類正確但位置錯亂 —— 單色服裝看不出來，雙色服裝才會暴露。
    #
    # 注意上面算遮罩時是手動取樣 (1-v)，與烘焙慣例一致，不需要也不能翻。
    uv_tex = uv.copy()
    if args.tex_flip_v:
        uv_tex[:, 1] = 1.0 - uv_tex[:, 1]
    uv_t = torch.tensor(uv_tex, device="cuda").unsqueeze(0)
    msk_t = torch.tensor(vmask[:, None], device="cuda").unsqueeze(0)
    tex_t = torch.tensor(tex, device="cuda").unsqueeze(0)

    panel_w = int(round(W * PANEL_H / H))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (panel_w * 3, PANEL_H))

    n_total = min(len(frames), n_video)
    if args.limit:
        n_total = min(n_total, args.limit)

    n_ok = 0
    for i in range(n_total):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        pose, j3d, j2d = frames[i]
        panel_l = frame_bgr.copy()

        if pose is None:
            # 這一影格沒偵測到人，中間格就顯示原畫面
            panels = [panel_l, frame_bgr.copy(), np.zeros_like(frame_bgr)]
            writer.write(_compose(panels, panel_w, PANEL_H))
            continue

        # ── LBS：套用 ROMP 的姿態 ──
        pose = pose.reshape(-1, 3)
        fp = bind["full_pose"].copy()
        take = min(len(pose), N_SMPL_BODY)
        fp[:take] = pose[:take]
        # global_orient 要做相機空間 → 世界座標校正，body_pose 不用
        from scipy.spatial.transform import Rotation
        fp[0] = Rotation.from_matrix(
            R_CW @ Rotation.from_rotvec(fp[0]).as_matrix()).as_rotvec()

        G_new = fk(J_rest, fp, parents)
        T = np.einsum("jab,jbc->jac", G_new, G_bind_inv)     # (J,4,4)
        # skin_weights.npz 存的是每頂點前 4 大的關節索引與權重（glTF 上限），
        # 所以這裡用 gather 而非完整的 (V,J) 權重矩陣。
        blended = np.einsum("vk,vkab->vab", swt, T[sj])       # (V,4,4)
        v_posed = np.einsum("vab,vb->va", blended, v_h)[:, :3]

        # ── 對位：3D 相似變換 + 2D 縮放平移 ──
        J_new = G_new[:, :3, 3]
        xy = None
        if j3d is not None and len(j3d) >= 22:
            s, R, t = umeyama(J_new[:22], np.asarray(j3d)[:22])
            v_al = (s * (R @ v_posed.T).T + t)
            j_al = (s * (R @ J_new[:22].T).T + t)
            if j2d is not None and len(j2d) >= 22:
                d2 = np.asarray(j2d)[:22, :2]
                if np.abs(d2).max() <= 1.5:          # 正規化座標
                    d2 = (d2 * 0.5 + 0.5) * np.array([W, H])
                s2, t2 = fit_2d(j_al[:, :2], d2)
                xy = v_al[:, :2] * s2 * args.scale + t2
                z = v_al[:, 2]
        if xy is None:
            # 沒有 3D/2D 關節可用，退回只用 2D 關節做正交對位
            if j2d is None:
                panels = [panel_l, frame_bgr.copy(), np.zeros_like(frame_bgr)]
                writer.write(_compose(panels, panel_w, PANEL_H))
                continue
            d2 = np.asarray(j2d)[:22, :2]
            if np.abs(d2).max() <= 1.5:
                d2 = (d2 * 0.5 + 0.5) * np.array([W, H])
            s2, t2 = fit_2d(J_new[:22, :2], d2)
            xy = v_posed[:, :2] * s2 * args.scale + t2
            z = v_posed[:, 2]

        xy = xy + np.array([args.x_shift, args.y_shift])

        # ── 光柵化 ──
        clip = np.zeros((len(xy), 4), np.float32)
        clip[:, 0] = (xy[:, 0] / W) * 2 - 1
        # 不要對 y 取負。xy 是影像像素座標（y 向下），而 nvdiffrast 輸出的
        # 第 0 列對應 clip y = -1，兩者方向一致，多取一次負號會讓人上下顛倒。
        clip[:, 1] = (xy[:, 1] / H) * 2 - 1
        zr = max(z.max() - z.min(), 1e-6)
        clip[:, 2] = (z - z.min()) / zr * 0.9 - 0.45
        clip[:, 3] = 1.0

        rgb_f, a_cloth, a_full = _raster(
            ctx, torch.tensor(clip, device="cuda").unsqueeze(0),
            faces_t, uv_t, msk_t, tex_t, H, W)

        # ── 中間格：只把衣物區域疊回原畫面 ──
        rgb = cv2.cvtColor((rgb_f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        a = a_cloth * args.alpha
        mid = (frame_bgr * (1 - a) + rgb * a).astype(np.uint8)

        # ── 右格 ──
        if args.debug_mask:
            # 疊合遮罩：白=會蓋住原畫面、灰=網格有但不疊合、黑=沒有網格。
            # 頭與手應該是灰色，如果是白色代表遮罩範圍錯了。
            m = (a_cloth * 255).astype(np.uint8)
            body = ((a_full - a_cloth) * 90).astype(np.uint8)
            right = cv2.cvtColor(np.clip(m + body, 0, 255)[..., 0],
                                 cv2.COLOR_GRAY2BGR)
        else:
            # 完整的著衣人體（不套衣物遮罩，含頭手鞋）
            right = _fit_avatar(rgb, a_full, frame_bgr.shape)

        writer.write(_compose([panel_l, mid, right], panel_w, PANEL_H))
        n_ok += 1
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{n_total}")

    cap.release()
    writer.release()
    print(f"\n[DONE] {args.out}（{n_ok}/{n_total} 影格成功疊合）")
    print("       左=原始影片　中=衣服疊回身上　右=3D 著衣模型")


def _raster(ctx, clip, faces_t, uv_t, msk_t, tex_t, H, W):
    """光柵化一次，回傳 (rgb, alpha_衣物, alpha_全身)。

    兩個 alpha 共用同一次光柵化：中間格要只顯示衣物（頭手透明，露出真人），
    右格要顯示完整著衣人體，差別只在遮罩，不必渲染兩次。
    衣物 alpha 用插值後的遮罩值，邊界天然帶柔邊。
    """
    import torch
    import nvdiffrast.torch as dr
    rast, _ = dr.rasterize(ctx, clip, faces_t, resolution=[H, W])
    uvi, _ = dr.interpolate(uv_t, rast, faces_t)
    mki, _ = dr.interpolate(msk_t, rast, faces_t)
    col = dr.texture(tex_t, uvi, filter_mode="linear")
    vis = (rast[..., 3:4] > 0).float()
    rgb = col[0].detach().cpu().numpy()
    a_cloth = (mki * vis)[0].detach().cpu().numpy()
    a_full = vis[0].detach().cpu().numpy()
    return rgb, a_cloth, a_full


def _fit_avatar(rgb_bgr, alpha, shape, pad=0.08):
    """把完整著衣人體裁切置中放進一格畫面。

    人體在原始畫面裡通常只佔一小塊，直接輸出右格會是一個很小的人。
    這裡依實際輪廓裁切再放大，讓右格看起來是專門展示模型的視圖。
    """
    H, W = shape[:2]
    out = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.where(alpha[..., 0] > 0.01)
    if len(ys) == 0:
        return out

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    m = int(pad * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - m), max(0, x0 - m)
    y1, x1 = min(H, y1 + m), min(W, x1 + m)

    crop = (rgb_bgr[y0:y1, x0:x1] * alpha[y0:y1, x0:x1]).astype(np.uint8)
    ch, cw = crop.shape[:2]
    s = min(H / ch, W / cw)
    nh, nw = max(1, int(ch * s)), max(1, int(cw * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = crop
    return out

r"""
render_overlay.py

把生成的著衣網格「穿」回原始影片裡的人身上，輸出三格對照影片：

    左：原始影片        中：衣服疊回使用者身上        右：3D 著衣模型

只重疊上衣與下著，頭、手、鞋維持原始畫面，所以看起來是使用者本人穿上生成的衣服。

在 chupa 環境執行（需要 nvdiffrast / smplx / trimesh / opencv）。

────────────────────────────────────────────────────────────────
運作方式
────────────────────────────────────────────────────────────────

1. 用 ROMP 每影格的姿態做 LBS，把著衣網格擺成該影格的姿勢。

2. 對位：把我們的 SMPL-X 關節與 ROMP 的 3D 關節做相似變換擬合
   （Umeyama，求縮放 + 旋轉 + 平移），把網格搬到 ROMP 的座標系。
   接著用 ROMP 的 2D 關節（pj2d_org，原圖像素座標）再擬合一次
   2D 的縮放與平移，確保投影後確實落在影片中的人身上。

   為什麼分兩段：ROMP 的相機是弱透視，等同正交投影加縮放，
   所以「3D 對齊 + 2D 擬合」比反推相機內參穩健得多，
   也不必假設 ROMP 各版本的相機參數格式一致。

3. 以 nvdiffrast 光柵化，用衣物遮罩當 alpha 疊回原始畫面。
   遮罩來自 semantic stage 產生的 label_atlas，插值後的邊界值
   直接當作柔邊 alpha，避免鋸齒。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  conda activate chupa
  cd ~/projects/chupa

  # 先看 ROMP 輸出有哪些欄位可用（1 秒，強烈建議先跑）
  python render_overlay.py --inspect --motion ~/projects/romp_test/img4640_output

  # 先渲染前 30 影格確認對位（約 1 分鐘）
  python render_overlay.py \
      --video ~/projects/romp_test/img4640.mp4 \
      --motion ~/projects/romp_test/img4640_output \
      --work tex_work_seed \
      --texture tex_work_seed/final/chupa_textured_texture.png \
      --bind-pkl data/renderpeople/smplx/rp_emily_apose_001.pkl \
      --limit 30 --out /tmp/overlay_test.mp4

  # 確認沒問題後跑完整段
  python render_overlay.py ... --out /tmp/overlay_full.mp4

────────────────────────────────────────────────────────────────
對位不準時怎麼調
────────────────────────────────────────────────────────────────
  衣服整體偏移      → 檢查 --motion 的 pj2d_org 是否為原圖座標
                       （某些版本是正規化的 [-1,1]，用 --inspect 看數值範圍）
  衣服比人大或小    → ALIGN_SCALE 微調
  衣服上下顛倒      → 對位用的關節對應錯了，看 --inspect 的關節數
  衣服穿在別人身上  → 影片裡有多人，用 --person 指定
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# 參數
# ============================================================

SMPLX_MODEL_DIR = "src/smpl_related/models"
NUM_JOINTS = 55
NUM_BODY_JOINTS = 21
N_SMPL_BODY = 22

R_CW = np.diag([1.0, -1.0, -1.0])       # ROMP 相機空間 → SMPL-X 世界座標

# 疊合哪些語意區域。1=上衣 2=下著 3=頭 4=手 5=鞋（見 uv_texture_bake.py）
OVERLAY_REGIONS = [1, 2]

ALIGN_SCALE = 1.0        # 衣服比人大就調小，例如 0.97
PANEL_H = 720            # 每格畫面高度


# ============================================================
# SMPL-X 前向運動學（與 chupa_reskin.py 一致）
# ============================================================

def rodrigues(rv):
    rv = np.asarray(rv, np.float64).reshape(-1, 3)
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    safe = np.where(th < 1e-8, 1.0, th)
    ax = rv / safe
    c, s = np.cos(th)[:, 0], np.sin(th)[:, 0]
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    R = np.zeros((len(rv), 3, 3))
    R[:, 0, 0] = c + x * x * (1 - c); R[:, 0, 1] = x * y * (1 - c) - z * s
    R[:, 0, 2] = x * z * (1 - c) + y * s; R[:, 1, 0] = y * x * (1 - c) + z * s
    R[:, 1, 1] = c + y * y * (1 - c); R[:, 1, 2] = y * z * (1 - c) - x * s
    R[:, 2, 0] = z * x * (1 - c) - y * s; R[:, 2, 1] = z * y * (1 - c) + x * s
    R[:, 2, 2] = c + z * z * (1 - c)
    R[th[:, 0] < 1e-8] = np.eye(3)
    return R


def fk(J_rest, full_pose, parents):
    """回傳每個關節的全域變換 (J,4,4)。"""
    R = rodrigues(full_pose.reshape(-1, 3))
    G = np.zeros((len(J_rest), 4, 4))
    G[0] = np.eye(4); G[0][:3, :3] = R[0]; G[0][:3, 3] = J_rest[0]
    for j in range(1, len(J_rest)):
        L = np.eye(4)
        L[:3, :3] = R[j]
        L[:3, 3] = J_rest[j] - J_rest[parents[j]]
        G[j] = G[parents[j]] @ L
    return G


def make_full_pose(go, bp63):
    fp = np.zeros((NUM_JOINTS, 3), np.float64)
    fp[0] = np.asarray(go).reshape(3)
    fp[1:1 + NUM_BODY_JOINTS] = np.asarray(bp63).reshape(NUM_BODY_JOINTS, 3)
    return fp


def load_bind(pkl_path):
    with open(pkl_path, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    go = np.array(p["global_orient"], np.float64).reshape(3)
    betas = np.array(p["betas"], np.float64).reshape(-1)
    bp = np.array(p["body_pose"], np.float64).reshape(-1, 3)
    bp63 = bp[1:22].reshape(63)          # 丟棄 slot 0
    return {"betas": betas,
            "full_pose": make_full_pose(go, bp63),
            "scale": float(np.asarray(p["body_scale"]).reshape(-1)[0]),
            "transl": np.asarray(p["global_body_translation"], np.float64).reshape(3)}


def build_smplx(model_dir, betas):
    import torch, smplx
    from smplx.lbs import vertices2joints, blend_shapes
    m = smplx.create(model_dir, model_type="smplx", gender="neutral",
                     use_pca=True, num_pca_comps=6, num_betas=10,
                     num_expression_coeffs=10, batch_size=1)
    m.eval()
    bt = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    v_shaped = m.v_template.unsqueeze(0) + blend_shapes(
        bt, m.shapedirs[:, :, :bt.shape[-1]])
    J = vertices2joints(m.J_regressor, v_shaped).squeeze(0).detach().numpy().astype(np.float64)
    return J, m.parents.detach().cpu().numpy()


# ============================================================
# ROMP 輸出
# ============================================================

POSE_KEYS = ["smpl_thetas", "poses", "pose"]
J3D_KEYS = ["joints", "j3d", "smpl_joints", "joints_smpl"]
J2D_KEYS = ["pj2d_org", "pj2d", "joints_2d", "kp2d"]


def _unwrap(d):
    if isinstance(d, dict) and "results" in d:
        r = d["results"]
        if isinstance(r, np.ndarray):
            r = r.item() if r.ndim == 0 else r[0]
        if isinstance(r, dict):
            return r
    return d


def _pick(d, keys):
    for k in keys:
        if k in d:
            return k, np.asarray(d[k], np.float64)
    return None, None


def load_romp_frames(path):
    """回傳 [(pose72, j3d, j2d), ...]，依影格順序。缺的欄位為 None。"""
    path = Path(path)
    if path.is_dir():
        summary = path / "video_results.npz"
        if summary.exists():
            path = summary
        else:
            files = sorted(f for f in path.glob("*.npz")
                           if not f.name.startswith("video_results"))
            return [_extract(_unwrap(dict(np.load(f, allow_pickle=True))))
                    for f in files]

    raw = _unwrap(dict(np.load(path, allow_pickle=True)))
    # 彙總檔：鍵是影格圖檔名
    ks = list(raw.keys())
    if ks and all(isinstance(k, str) for k in ks) and \
       any(k.lower().endswith((".png", ".jpg")) for k in ks[:3]):
        return [_extract(raw[k]) for k in sorted(ks)]
    return [_extract(raw)]


def _extract(d, person=0):
    if isinstance(d, np.ndarray):
        d = d.item() if d.ndim == 0 else d[0]
    d = _unwrap(d)

    def take(keys, dim):
        k, v = _pick(d, keys)
        if v is None:
            return None
        v = v.reshape(-1, *v.shape[1:]) if v.ndim > dim else v[None]
        if len(v) == 0:
            return None
        return v[min(person, len(v) - 1)]

    return (take(POSE_KEYS, 1), take(J3D_KEYS, 2), take(J2D_KEYS, 2))


def inspect(path):
    """把 ROMP 輸出的欄位與數值範圍印出來 —— 對位邏輯依賴這些格式。"""
    path = Path(path)
    f = path / "video_results.npz" if path.is_dir() else path
    if not f.exists():
        cands = sorted(path.glob("*.npz"))
        f = cands[0] if cands else None
    if f is None:
        print(f"找不到 npz: {path}")
        return

    raw = _unwrap(dict(np.load(f, allow_pickle=True)))
    ks = list(raw.keys())
    if ks and any(str(k).lower().endswith((".png", ".jpg")) for k in ks[:3]):
        print(f"彙總檔，{len(ks)} 影格。取第一格檢視：")
        d = raw[sorted(ks)[0]]
        d = d.item() if isinstance(d, np.ndarray) and d.ndim == 0 else d
        d = _unwrap(d)
    else:
        d = raw

    print(f"\n檔案: {f}")
    print(f"欄位: {list(d.keys())}\n")
    for k, v in d.items():
        v = np.asarray(v)
        if v.dtype.kind in "fiu" and v.size:
            print(f"  {k:<16} shape={str(v.shape):<16} "
                  f"範圍 [{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k:<16} shape={v.shape} dtype={v.dtype}")
    print("\n要用到的欄位：")
    for name, keys in [("姿態", POSE_KEYS), ("3D 關節", J3D_KEYS), ("2D 關節", J2D_KEYS)]:
        k, _ = _pick(d, keys)
        print(f"  {name:<8}: {k or '找不到 —— 對位會失敗，請回報這份清單'}")
    print("\n注意：2D 關節若範圍在 [-1,1] 是正規化座標，需乘上影像尺寸；"
          "若範圍是幾百到上千則已是像素座標。")


# ============================================================
# 對位
# ============================================================

def umeyama(src, dst):
    """求相似變換 (scale, R, t) 使 scale*R@src + t ≈ dst。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    C = d0.T @ s0 / len(src)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (s0 ** 2).sum() / len(src)
    scale = np.trace(np.diag(S) @ D) / max(var, 1e-12)
    return scale, R, mu_d - scale * R @ mu_s


def fit_2d(src_xy, dst_xy):
    """求 2D 的等向縮放與平移，使 s*src + t ≈ dst。"""
    mu_s, mu_d = src_xy.mean(0), dst_xy.mean(0)
    s0, d0 = src_xy - mu_s, dst_xy - mu_d
    denom = (s0 ** 2).sum()
    s = float((s0 * d0).sum() / max(denom, 1e-12))
    return s, mu_d - s * mu_s


# ============================================================
# 主流程
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", help="原始影片檔")
    p.add_argument("--motion", required=True, help="ROMP 輸出資料夾或彙總檔")
    p.add_argument("--work", default="tex_work_seed", help="含 geometry/label/weights 的資料夾")
    p.add_argument("--texture", default=None)
    p.add_argument("--bind-pkl", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    p.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR)
    p.add_argument("--out", default="/tmp/overlay.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0, help="只處理前 N 影格（測試用）")
    p.add_argument("--person", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0, help="疊合不透明度")
    p.add_argument("--regions", type=int, nargs="+", default=OVERLAY_REGIONS,
                   help="要疊合的語意區域。1=上衣 2=下著 3=頭 4=手 5=鞋")
    p.add_argument("--scale", type=float, default=ALIGN_SCALE,
                   help="整體縮放微調。衣服比人大就調小，例如 0.95")
    p.add_argument("--y-shift", type=float, default=0.0,
                   help="垂直微調（像素，正值往下）。衣領蓋到臉就調正值")
    p.add_argument("--x-shift", type=float, default=0.0, help="水平微調（像素）")
    p.add_argument("--debug-mask", action="store_true",
                   help="右格改顯示疊合遮罩（白=會被蓋住），用來確認遮罩範圍")
    p.add_argument("--no-tex-flip-v", dest="tex_flip_v", action="store_false",
                   help="關閉貼圖 V 軸翻轉。顏色種類正確但位置錯亂時切換看看")
    p.add_argument("--fit-avatar", action="store_true",
                   help="右格裁切放大讓模型佔滿畫面。預設維持與影片相同大小")
    p.set_defaults(tex_flip_v=True)
    p.add_argument("--inspect", action="store_true", help="只檢視 ROMP 輸出格式")
    args = p.parse_args()

    if args.inspect:
        inspect(args.motion)
        return

    if not args.video:
        p.error("需要 --video（或用 --inspect 檢視格式）")

    import torch
    import nvdiffrast.torch as dr

    work = Path(args.work)
    g = np.load(work / "geometry.npz")
    verts_w, faces, uv = (g["verts"].astype(np.float64),
                          g["faces"].astype(np.int32),
                          g["uv"].astype(np.float32))
    sw = np.load(work / "skin_weights.npz")
    sj, swt = sw["joints"].astype(np.int64), sw["weights"].astype(np.float64)

    label = np.load(work / "label_atlas.npy")
    tex_path = args.texture or str(work / "texture.png")
    tex = cv2.cvtColor(cv2.imread(tex_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 衣物遮罩：先在 UV 空間標出要疊合的區域，再依每個頂點的 UV 取樣成頂點屬性。
    # 用頂點屬性而非在貼圖空間做，光柵化時插值出來的邊界值可以直接當柔邊 alpha。
    res = label.shape[0]
    tx = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
    ty = np.clip(((1 - uv[:, 1]) * res).astype(int), 0, res - 1)
    # 把標籤填滿 xatlas 的打包間隙後再取樣。
    #
    # 頂點的 UV 換算成整數 texel 時，位於 UV 島邊界的頂點常會落在島與島之間的
    # 間隙上（標籤 0），被判定為「不疊合」，結果是衣服上佈滿細小的破洞，
    # 原始影片會從洞裡透出來。貼圖本身有做邊界外擴所以顏色正常，
    # 但標籤圖是離散值、沒有做外擴，必須在這裡補。
    #
    # 用最近非零標籤填補（而非形態學膨脹），區域交界處才不會偏向某一邊。
    from scipy.ndimage import distance_transform_edt
    if (label == 0).any():
        _, (iy, ix) = distance_transform_edt(label == 0, return_indices=True)
        label_filled = label[iy, ix]
        n_gap = int((label == 0).sum())
        print(f"[INFO] 已填補標籤圖的打包間隙（{n_gap} texel）")
    else:
        label_filled = label
    label = label_filled

    sampled = label[ty, tx]
    vmask = np.isin(sampled, args.regions).astype(np.float32)
    names = {0: "(間隙)", 1: "上衣", 2: "下著", 3: "頭", 4: "手", 5: "鞋"}
    print(f"[INFO] 頂點的語意區域分布：")
    for rid in sorted(set(sampled.tolist())):
        n = int((sampled == rid).sum())
        mark = " ← 疊合" if rid in args.regions else ""
        print(f"       {names.get(rid, rid):<8} {n:>7} ({100*n/len(sampled):5.1f}%){mark}")
    print(f"[INFO] 疊合區域 {args.regions}，佔 {100 * vmask.mean():.1f}% 頂點")

    bind = load_bind(args.bind_pkl)
    J_rest, parents = build_smplx(args.smplx_model_dir, bind["betas"])
    G_bind = fk(J_rest, bind["full_pose"], parents)
    G_bind_inv = np.linalg.inv(G_bind)
    v_raw = (verts_w - bind["transl"].reshape(1, 3)) / bind["scale"]
    v_h = np.concatenate([v_raw, np.ones((len(v_raw), 1))], axis=1)

    frames = load_romp_frames(args.motion)
    print(f"[INFO] ROMP 影格數: {len(frames)}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 影片 {W}x{H}, {n_video} 影格")
    if abs(n_video - len(frames)) > 5:
        print(f"[WARN] 影片影格數與 ROMP 影格數差距較大，"
              f"對位會逐格套用，數量不符的部分會被截斷")

    ctx = dr.RasterizeCudaContext()
    faces_t = torch.tensor(faces, dtype=torch.int32, device="cuda")

    # 貼圖採樣用的 UV 要翻轉 V 軸。
    #
    # 烘貼圖時 texel 的列是用 (1-v)*RES 算的（v=1 對應第 0 列），
    # 但 nvdiffrast 的 dr.texture() 把 v 直接當成列的比例（v=0 對應第 0 列），
    # 兩者相反。不翻轉的話每個 UV 島會取到鏡射位置的內容，
    # 症狀是顏色種類正確但位置錯亂 —— 單色服裝看不出來，雙色服裝才會暴露。
    #
    # 注意上面算遮罩時是手動取樣 (1-v)，與烘焙慣例一致，不需要也不能翻。
    uv_tex = uv.copy()
    if args.tex_flip_v:
        uv_tex[:, 1] = 1.0 - uv_tex[:, 1]
    uv_t = torch.tensor(uv_tex, device="cuda").unsqueeze(0)
    msk_t = torch.tensor(vmask[:, None], device="cuda").unsqueeze(0)
    tex_t = torch.tensor(tex, device="cuda").unsqueeze(0)

    panel_w = int(round(W * PANEL_H / H))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (panel_w * 3, PANEL_H))

    n_total = min(len(frames), n_video)
    if args.limit:
        n_total = min(n_total, args.limit)

    n_ok = 0
    for i in range(n_total):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        pose, j3d, j2d = frames[i]
        panel_l = frame_bgr.copy()

        if pose is None:
            # 這一影格沒偵測到人，中間格就顯示原畫面
            panels = [panel_l, frame_bgr.copy(), np.zeros_like(frame_bgr)]
            writer.write(_compose(panels, panel_w, PANEL_H))
            continue

        # ── LBS：套用 ROMP 的姿態 ──
        pose = pose.reshape(-1, 3)
        fp = bind["full_pose"].copy()
        take = min(len(pose), N_SMPL_BODY)
        fp[:take] = pose[:take]
        # global_orient 要做相機空間 → 世界座標校正，body_pose 不用
        from scipy.spatial.transform import Rotation
        fp[0] = Rotation.from_matrix(
            R_CW @ Rotation.from_rotvec(fp[0]).as_matrix()).as_rotvec()

        G_new = fk(J_rest, fp, parents)
        T = np.einsum("jab,jbc->jac", G_new, G_bind_inv)     # (J,4,4)
        # skin_weights.npz 存的是每頂點前 4 大的關節索引與權重（glTF 上限），
        # 所以這裡用 gather 而非完整的 (V,J) 權重矩陣。
        blended = np.einsum("vk,vkab->vab", swt, T[sj])       # (V,4,4)
        v_posed = np.einsum("vab,vb->va", blended, v_h)[:, :3]

        # ── 對位：3D 相似變換 + 2D 縮放平移 ──
        J_new = G_new[:, :3, 3]
        xy = None
        if j3d is not None and len(j3d) >= 22:
            s, R, t = umeyama(J_new[:22], np.asarray(j3d)[:22])
            v_al = (s * (R @ v_posed.T).T + t)
            j_al = (s * (R @ J_new[:22].T).T + t)
            if j2d is not None and len(j2d) >= 22:
                d2 = np.asarray(j2d)[:22, :2]
                if np.abs(d2).max() <= 1.5:          # 正規化座標
                    d2 = (d2 * 0.5 + 0.5) * np.array([W, H])
                s2, t2 = fit_2d(j_al[:, :2], d2)
                xy = v_al[:, :2] * s2 * args.scale + t2
                z = v_al[:, 2]
        if xy is None:
            # 沒有 3D/2D 關節可用，退回只用 2D 關節做正交對位
            if j2d is None:
                panels = [panel_l, frame_bgr.copy(), np.zeros_like(frame_bgr)]
                writer.write(_compose(panels, panel_w, PANEL_H))
                continue
            d2 = np.asarray(j2d)[:22, :2]
            if np.abs(d2).max() <= 1.5:
                d2 = (d2 * 0.5 + 0.5) * np.array([W, H])
            s2, t2 = fit_2d(J_new[:22, :2], d2)
            xy = v_posed[:, :2] * s2 * args.scale + t2
            z = v_posed[:, 2]

        xy = xy + np.array([args.x_shift, args.y_shift])

        # ── 光柵化 ──
        clip = np.zeros((len(xy), 4), np.float32)
        clip[:, 0] = (xy[:, 0] / W) * 2 - 1
        # 不要對 y 取負。xy 是影像像素座標（y 向下），而 nvdiffrast 輸出的
        # 第 0 列對應 clip y = -1，兩者方向一致，多取一次負號會讓人上下顛倒。
        clip[:, 1] = (xy[:, 1] / H) * 2 - 1
        zr = max(z.max() - z.min(), 1e-6)
        clip[:, 2] = (z - z.min()) / zr * 0.9 - 0.45
        clip[:, 3] = 1.0

        rgb_f, a_cloth, a_full = _raster(
            ctx, torch.tensor(clip, device="cuda").unsqueeze(0),
            faces_t, uv_t, msk_t, tex_t, H, W)

        # ── 中間格：只把衣物區域疊回原畫面 ──
        rgb = cv2.cvtColor((rgb_f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        a = a_cloth * args.alpha
        mid = (frame_bgr * (1 - a) + rgb * a).astype(np.uint8)

        # ── 右格 ──
        if args.debug_mask:
            # 疊合遮罩：白=會蓋住原畫面、灰=網格有但不疊合、黑=沒有網格。
            # 頭與手應該是灰色，如果是白色代表遮罩範圍錯了。
            m = (a_cloth * 255).astype(np.uint8)
            body = ((a_full - a_cloth) * 90).astype(np.uint8)
            right = cv2.cvtColor(np.clip(m + body, 0, 255)[..., 0],
                                 cv2.COLOR_GRAY2BGR)
        elif args.fit_avatar:
            # 裁切放大版：右格專門展示模型，人會佔滿畫面
            right = _fit_avatar(rgb, a_full, frame_bgr.shape)
        else:
            # 預設：維持與影片相同的投影，所以三格的人是同樣大小、同樣位置，
            # 並排比對時視線不用重新對焦
            right = (rgb * a_full).astype(np.uint8)

        writer.write(_compose([panel_l, mid, right], panel_w, PANEL_H))
        n_ok += 1
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{n_total}")

    cap.release()
    writer.release()
    print(f"\n[DONE] {args.out}（{n_ok}/{n_total} 影格成功疊合）")
    print("       左=原始影片　中=衣服疊回身上　右=3D 著衣模型")


def _raster(ctx, clip, faces_t, uv_t, msk_t, tex_t, H, W):
    """光柵化一次，回傳 (rgb, alpha_衣物, alpha_全身)。

    兩個 alpha 共用同一次光柵化：中間格要只顯示衣物（頭手透明，露出真人），
    右格要顯示完整著衣人體，差別只在遮罩，不必渲染兩次。
    衣物 alpha 用插值後的遮罩值，邊界天然帶柔邊。
    """
    import torch
    import nvdiffrast.torch as dr
    rast, _ = dr.rasterize(ctx, clip, faces_t, resolution=[H, W])
    uvi, _ = dr.interpolate(uv_t, rast, faces_t)
    mki, _ = dr.interpolate(msk_t, rast, faces_t)
    col = dr.texture(tex_t, uvi, filter_mode="linear")
    vis = (rast[..., 3:4] > 0).float()
    rgb = col[0].detach().cpu().numpy()
    a_cloth = (mki * vis)[0].detach().cpu().numpy()
    a_full = vis[0].detach().cpu().numpy()
    return rgb, a_cloth, a_full


def _fit_avatar(rgb_bgr, alpha, shape, pad=0.08):
    """把完整著衣人體裁切置中放進一格畫面。

    人體在原始畫面裡通常只佔一小塊，直接輸出右格會是一個很小的人。
    這裡依實際輪廓裁切再放大，讓右格看起來是專門展示模型的視圖。
    """
    H, W = shape[:2]
    out = np.zeros((H, W, 3), np.uint8)
    ys, xs = np.where(alpha[..., 0] > 0.01)
    if len(ys) == 0:
        return out

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    m = int(pad * max(y1 - y0, x1 - x0))
    y0, x0 = max(0, y0 - m), max(0, x0 - m)
    y1, x1 = min(H, y1 + m), min(W, x1 + m)

    crop = (rgb_bgr[y0:y1, x0:x1] * alpha[y0:y1, x0:x1]).astype(np.uint8)
    ch, cw = crop.shape[:2]
    s = min(H / ch, W / cw)
    nh, nw = max(1, int(ch * s)), max(1, int(cw * s))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = crop
    return out


def _compose(panels, w, h):
    out = []
    for p in panels:
        out.append(cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA))
    return np.hstack(out)


if __name__ == "__main__":
    main()
def _compose(panels, w, h):
    out = []
    for p in panels:
        out.append(cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA))
    return np.hstack(out)


if __name__ == "__main__":
    main()