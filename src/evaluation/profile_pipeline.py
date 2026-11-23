r"""
profile_pipeline.py

逐階段量測每影格耗時，輸出含 Mean / Std / P95 / 佔比的表格。

因 ROMP 與光柵化分屬兩個互不相容的 Python 環境，量測分兩段執行後再合併：

    --part romp      於 romp 環境：影像前處理、ROMP 推論
    --part render    於 chupa 環境：姿態轉換、FK、LBS、光柵化／貼圖、Alpha 合成
    --part endtoend  單一行程的端到端量測（前處理→ROMP→轉換→FK→LBS）
                     需同時具備 romp 與 smplx；所得為真正的端到端延遲
    --report         合併結果，輸出 Markdown 表格（含環境內部之真 P95）

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  # 第一段（romp 環境）
  conda activate romp
  cd ~/projects/chupa
  python profile_pipeline.py --part romp \
      --video ~/projects/romp_test/IMG_4640.mov \
      --out prof_romp.json

  # 第二段（chupa 環境）
  conda activate chupa
  python profile_pipeline.py --part render \
      --video ~/projects/romp_test/IMG_4640.mov \
      --motion ~/projects/romp_test/img4640_output \
      --work demo_work/A2_navy_coat \
      --bind-pkl data/renderpeople/smplx/rp_emily_apose_001.pkl \
      --out prof_render.json

  # 合併出表
  python profile_pipeline.py --report prof_romp.json prof_render.json \
      --out timing_table.md

────────────────────────────────────────────────────────────────
量測上的注意事項
────────────────────────────────────────────────────────────────

* **暖機影格必須排除。** 前幾影格包含 CUDA 上下文建立、核心編譯與快取填充，
  其耗時可達穩態的數倍。本腳本預設丟棄前 20 影格（--warmup）。

* **GPU 為非同步執行。** 若不強制同步，計時只會量到「把工作丟給 GPU」的時間，
  而非實際完成的時間。本腳本於每個 GPU 階段後呼叫 torch.cuda.synchronize()，
  故所測為真實的牆鐘時間。此舉會略微增加總耗時，但這是取得可信分階段數據的
  必要代價——若不同步，各階段的數字會失真且彼此無法相加。

* **兩段量測不可直接相加為端到端延遲。** 實際系統中兩段於不同行程執行，
  尚有行程間傳遞的開銷；表中的「完整單幀」為各階段之和，代表計算成本的
  下界，而非實際端到端延遲。
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np


STAGE_ORDER = [
    ("preprocess", "影像前處理"),
    ("romp",       "ROMP"),
    ("convert",    "SMPL→SMPL-X"),
    ("fk",         "FK"),
    ("lbs",        "LBS"),
    ("raster",     "光柵化／貼圖"),
    ("composite",  "Alpha compositing"),
]

NUM_JOINTS = 55
NUM_BODY_JOINTS = 21
R_CW = np.diag([1.0, -1.0, -1.0])


class Timer:
    """累積各階段的逐影格耗時。sync 用於在 GPU 階段後強制同步。"""

    def __init__(self, sync=None):
        self.rec = {}
        self.sync = sync

    def __call__(self, name):
        return _Scope(self, name)

    def add(self, name, dt):
        self.rec.setdefault(name, []).append(dt)

    def drop_warmup(self, n):
        for k in self.rec:
            self.rec[k] = self.rec[k][n:]

    def dump(self, path, meta):
        out = {"meta": meta, "stages": {k: v for k, v in self.rec.items()}}
        Path(path).write_text(json.dumps(out), encoding="utf-8")
        print(f"\n已寫出 {path}")


class _Scope:
    def __init__(self, timer, name):
        self.t, self.n = timer, name

    def __enter__(self):
        if self.t.sync:
            self.t.sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.t.sync:
            self.t.sync()
        self.t.add(self.n, time.perf_counter() - self.t0)
        return False


# ============================================================
# 共用：SMPL-X 前向運動學
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
    R = rodrigues(full_pose.reshape(-1, 3))
    G = np.zeros((len(J_rest), 4, 4))
    G[0] = np.eye(4); G[0][:3, :3] = R[0]; G[0][:3, 3] = J_rest[0]
    for j in range(1, len(J_rest)):
        L = np.eye(4)
        L[:3, :3] = R[j]
        L[:3, 3] = J_rest[j] - J_rest[parents[j]]
        G[j] = G[parents[j]] @ L
    return G


def load_bind(pkl_path):
    with open(pkl_path, "rb") as f:
        p = pickle.load(f, encoding="latin1")
    bp = np.array(p["body_pose"], np.float64).reshape(-1, 3)
    fp = np.zeros((NUM_JOINTS, 3), np.float64)
    fp[0] = np.array(p["global_orient"], np.float64).reshape(3)
    fp[1:1 + NUM_BODY_JOINTS] = bp[1:22]
    return {"betas": np.array(p["betas"], np.float64).reshape(-1),
            "full_pose": fp,
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
    v = m.v_template.unsqueeze(0) + blend_shapes(bt, m.shapedirs[:, :, :bt.shape[-1]])
    J = vertices2joints(m.J_regressor, v).squeeze(0).detach().numpy().astype(np.float64)
    return J, m.parents.detach().cpu().numpy()


# ============================================================
# 第一段：romp 環境
# ============================================================

def part_romp(args):
    import cv2
    import romp

    print("初始化 ROMP ...")
    settings = romp.main.default_settings
    settings.calc_smpl = True
    model = romp.ROMP(settings)

    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    except ImportError:
        sync = None

    t = Timer(sync)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片：{args.video}")

    n = 0
    while n < args.frames + args.warmup:
        with t("preprocess"):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with t("romp"):
            _ = model(rgb)

        n += 1
        if n % 50 == 0:
            print(f"  {n} 影格")
    cap.release()

    t.drop_warmup(args.warmup)
    print(f"有效影格 {len(t.rec.get('romp', []))}（已丟棄暖機 {args.warmup} 影格）")
    t.dump(args.out, {"part": "romp", "video": args.video,
                      "warmup": args.warmup, "gpu_sync": sync is not None})


# ============================================================
# 第二段：chupa 環境
# ============================================================

def part_render(args):
    import cv2
    import torch
    import nvdiffrast.torch as dr
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import distance_transform_edt

    sync = torch.cuda.synchronize if torch.cuda.is_available() else None
    t = Timer(sync)

    work = Path(args.work)
    g = np.load(work / "geometry.npz")
    verts_w = g["verts"].astype(np.float64)
    faces = g["faces"].astype(np.int32)
    uv = g["uv"].astype(np.float32)
    sw = np.load(work / "skin_weights.npz")
    sj, swt = sw["joints"].astype(np.int64), sw["weights"].astype(np.float64)

    label = np.load(work / "label_atlas.npy")
    if (label == 0).any():
        _, (iy, ix) = distance_transform_edt(label == 0, return_indices=True)
        label = label[iy, ix]
    tex = cv2.cvtColor(cv2.imread(str(work / "texture.png")),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    res = label.shape[0]
    tx_i = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
    ty_i = np.clip(((1 - uv[:, 1]) * res).astype(int), 0, res - 1)
    vmask = np.isin(label[ty_i, tx_i], [1, 2]).astype(np.float32)

    bind = load_bind(args.bind_pkl)
    J_rest, parents = build_smplx(args.smplx_model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(fk(J_rest, bind["full_pose"], parents))
    v_raw = (verts_w - bind["transl"].reshape(1, 3)) / bind["scale"]
    v_h = np.concatenate([v_raw, np.ones((len(v_raw), 1))], axis=1)

    # 動作資料（沿用 render_overlay 的載入邏輯）
    import importlib.util
    spec = importlib.util.spec_from_file_location("ro", "render_overlay.py")
    ro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ro)
    frames = ro.load_romp_frames(args.motion)
    print(f"動作影格 {len(frames)}")

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ctx = dr.RasterizeCudaContext()
    faces_t = torch.tensor(faces, dtype=torch.int32, device="cuda")
    uv_tex = uv.copy(); uv_tex[:, 1] = 1.0 - uv_tex[:, 1]
    uv_t = torch.tensor(uv_tex, device="cuda").unsqueeze(0)
    msk_t = torch.tensor(vmask[:, None], device="cuda").unsqueeze(0)
    tex_t = torch.tensor(tex, device="cuda").unsqueeze(0)

    n = 0
    limit = min(len(frames), args.frames + args.warmup)
    while n < limit:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        pose, j3d, j2d = frames[n]
        if pose is None or j3d is None or j2d is None:
            n += 1
            continue

        with t("convert"):
            pose = pose.reshape(-1, 3)
            fp = bind["full_pose"].copy()
            take = min(len(pose), 22)
            fp[:take] = pose[:take]
            fp[0] = Rotation.from_matrix(
                R_CW @ Rotation.from_rotvec(fp[0]).as_matrix()).as_rotvec()

        with t("fk"):
            G_new = fk(J_rest, fp, parents)
            T = np.einsum("jab,jbc->jac", G_new, G_bind_inv)

        with t("lbs"):
            blended = np.einsum("vk,vkab->vab", swt, T[sj])
            v_posed = np.einsum("vab,vb->va", blended, v_h)[:, :3]

        with t("raster"):
            J_new = G_new[:, :3, 3]
            s, R, tr = ro.umeyama(J_new[:22], np.asarray(j3d)[:22])
            v_al = (s * (R @ v_posed.T).T + tr)
            j_al = (s * (R @ J_new[:22].T).T + tr)
            d2 = np.asarray(j2d)[:22, :2]
            if np.abs(d2).max() <= 1.5:
                d2 = (d2 * 0.5 + 0.5) * np.array([W, H])
            s2, t2 = ro.fit_2d(j_al[:, :2], d2)
            xy = v_al[:, :2] * s2 + t2
            z = v_al[:, 2]

            clip = np.zeros((len(xy), 4), np.float32)
            clip[:, 0] = (xy[:, 0] / W) * 2 - 1
            clip[:, 1] = (xy[:, 1] / H) * 2 - 1
            zr = max(z.max() - z.min(), 1e-6)
            clip[:, 2] = (z - z.min()) / zr * 0.9 - 0.45
            clip[:, 3] = 1.0
            rgb_f, a_cloth, a_full = ro._raster(
                ctx, torch.tensor(clip, device="cuda").unsqueeze(0),
                faces_t, uv_t, msk_t, tex_t, H, W)

        with t("composite"):
            rgb = cv2.cvtColor((rgb_f * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            _ = (frame_bgr * (1 - a_cloth) + rgb * a_cloth).astype(np.uint8)

        n += 1
        if n % 50 == 0:
            print(f"  {n} 影格")
    cap.release()

    t.drop_warmup(args.warmup)
    print(f"有效影格 {len(t.rec.get('lbs', []))}（已丟棄暖機 {args.warmup} 影格）")
    t.dump(args.out, {"part": "render", "work": args.work,
                      "n_verts": int(len(verts_w)), "resolution": [W, H],
                      "warmup": args.warmup, "gpu_sync": sync is not None})


# ============================================================
# 合併報表
# ============================================================

def part_endtoend(args):
    r"""單一行程的端到端量測：影像前處理 → ROMP → 姿態轉換 → FK → LBS。

    **為何需要此模式。** `--part romp` 與 `--part render` 分屬兩個環境，
    無法對應同一影格，故其結果只能相加為「各階段平均之和」，代表計算成本
    而非端到端吞吐量；且各階段 P95 相加不等於端到端 P95。

    即時驅動路徑於實際系統中運行於單一環境，故其端到端量測是做得到的。
    本模式於同一行程內逐影格記錄**總牆鐘時間**，所得為真正的端到端延遲，
    且自然包含資料在各步驟間傳遞、記憶體配置與 Python 直譯器的開銷——
    這些正是分階段量測所排除、而使用者實際會感受到的成本。

    **執行環境需求**：同時具備 `romp` 與 `smplx`。若當前環境缺其一，
    請先安裝，或確認即時系統實際運行於哪個環境後於該環境執行。

    本模式**不含**光柵化與 Alpha 合成——該路徑目前實作為離線影片處理
    （論文 3.9 節），尚未納入即時迴圈，故其端到端量測須待整合後方能進行。
    """
    import cv2
    import romp
    import torch
    from scipy.spatial.transform import Rotation

    sync = torch.cuda.synchronize if torch.cuda.is_available() else None

    print("初始化 ROMP ...")
    settings = romp.main.default_settings
    settings.calc_smpl = True
    model = romp.ROMP(settings)

    work = Path(args.work)
    g = np.load(work / "geometry.npz")
    verts_w = g["verts"].astype(np.float64)
    sw = np.load(work / "skin_weights.npz")
    sj, swt = sw["joints"].astype(np.int64), sw["weights"].astype(np.float64)

    bind = load_bind(args.bind_pkl)
    J_rest, parents = build_smplx(args.smplx_model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(fk(J_rest, bind["full_pose"], parents))
    v_raw = (verts_w - bind["transl"].reshape(1, 3)) / bind["scale"]
    v_h = np.concatenate([v_raw, np.ones((len(v_raw), 1))], axis=1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片：{args.video}")

    t = Timer(sync)
    n = 0
    while n < args.frames + args.warmup:
        # 單一計時範圍涵蓋整個影格，含所有中間傳遞與配置成本
        with t("frame_total"):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out = model(rgb)
            if out is None or "smpl_thetas" not in out:
                n += 1
                continue
            pose = np.asarray(out["smpl_thetas"][0]).reshape(-1, 3)

            fp = bind["full_pose"].copy()
            take = min(len(pose), 22)
            fp[:take] = pose[:take]
            fp[0] = Rotation.from_matrix(
                R_CW @ Rotation.from_rotvec(fp[0]).as_matrix()).as_rotvec()

            G = fk(J_rest, fp, parents)
            T = np.einsum("jab,jbc->jac", G, G_bind_inv)
            M = np.einsum("nk,nkab->nab", swt, T[sj])
            _ = np.einsum("nab,nb->na", M, v_h)[:, :3]

        n += 1
        if n % 50 == 0:
            print(f"  {n} 影格")
    cap.release()

    t.drop_warmup(args.warmup)
    arr = np.asarray(t.rec.get("frame_total", [])) * 1000.0
    if len(arr) == 0:
        raise RuntimeError("沒有有效影格")

    print(f"\n端到端量測（單一行程，{len(arr)} 個有效影格）")
    print(f"  Mean {arr.mean():.2f} ms   Std {arr.std():.2f} ms")
    print(f"  P95  {np.percentile(arr, 95):.2f} ms   Max {arr.max():.2f} ms")
    print(f"  對應更新率：平均 {1000/arr.mean():.1f} FPS、"
          f"P95 {1000/np.percentile(arr, 95):.1f} FPS")
    print("\n此為真正的端到端延遲，含各步驟間的傳遞與配置開銷，"
          "可與分階段量測之和對照——兩者之差即為分階段量測所排除的部分。")

    t.dump(args.out, {"part": "endtoend", "video": args.video,
                      "work": str(work), "warmup": args.warmup,
                      "gpu_sync": sync is not None,
                      "note": "單一行程端到端；不含光柵化與 Alpha 合成"})


def env_subtotals(paths):
    r"""計算各執行環境內部的真實逐影格小計。

    **為何需要此函式。** 百分位數不具可加性：$P_{95}(X+Y) \neq P_{95}(X)+P_{95}(Y)$。
    將各階段的 P95 相加，隱含「所有階段於同一影格同時達到各自的第 95 百分位」，
    若各階段波動獨立，該情境的機率遠低於 5%，故其和為偏保守的高估。

    但同一個 JSON 內的各階段是**同一影格**量測的，故可逐影格相加後再取百分位，
    得到該環境內部真正的小計 P95。跨環境無法如此處理（無法對應同一影格），
    此即分階段量測架構的內在限制。

    回傳 [(檔名, 逐影格小計陣列, 該檔涵蓋的階段名), ...]
    """
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        st = d["stages"]
        if not st:
            continue
        n = min(len(v) for v in st.values())
        tot = np.zeros(n)
        for v in st.values():
            tot += np.asarray(v[:n])
        out.append((Path(p).name, tot * 1000.0, list(st.keys())))
    return out


def report(args):
    stages, meta = {}, []
    for p in args.report:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        stages.update(d["stages"])
        meta.append(d["meta"])

    rows, total_mean = [], 0.0
    for key, label in STAGE_ORDER:
        v = stages.get(key)
        if not v:
            rows.append((label, None))
            continue
        a = np.asarray(v) * 1000.0        # 毫秒
        rows.append((label, (a.mean(), a.std(), np.percentile(a, 95), len(a))))
        total_mean += a.mean()

    lines = ["| 階段 | Mean | Std. | P95 | 佔比 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, st in rows:
        if st is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        m, s, p95, _ = st
        lines.append(f"| {label} | {m:.2f} ms | {s:.2f} ms | {p95:.2f} ms | "
                     f"{100 * m / total_mean:.1f}% |")
    fps = 1000.0 / total_mean if total_mean else 0
    lines.append(f"| **完整單幀** | **{total_mean:.2f} ms** | — | — | **100%** |")

    n = min((st[3] for _, st in rows if st), default=0)
    txt = "\n".join(lines)
    txt += (f"\n\n各階段平均之和為 {total_mean:.2f} ms，對應等效更新率 {fps:.1f} FPS。"
            f"有效影格數 {n}。\n")

    # 環境內部的真 P95（逐影格相加後取百分位）
    subs = env_subtotals(args.report)
    if subs:
        txt += "\n## 各環境內部之真實小計（逐影格相加後取百分位）\n\n"
        txt += "| 量測檔 | 涵蓋階段 | Mean | P95 | Max |\n"
        txt += "| --- | --- | ---: | ---: | ---: |\n"
        for name, tot, keys in subs:
            txt += (f"| {name} | {len(keys)} 個階段 | {tot.mean():.2f} ms | "
                    f"{np.percentile(tot, 95):.2f} ms | {tot.max():.2f} ms |\n")

    sum_p95 = sum(st[2] for _, st in rows if st)
    txt += (f"\n**關於百分位數的加總。** 各階段 P95 之和為 {sum_p95:.2f} ms，"
            f"惟此值**不等同於端到端延遲的第 95 百分位**——百分位數不具可加性，"
            f"該和隱含所有階段於同一影格同時達到各自 P95 的情境，"
            f"若各階段波動獨立則其機率遠低於 5%，故為偏保守的估計。"
            f"上表之環境內部小計係逐影格相加後才取百分位，方為該環境內真正的 P95；"
            f"跨環境則因無法對應同一影格而無法計算。"
            f"另須注意 **P95 並非最壞情況**，其上仍有 5% 的影格；"
            f"上表另列出實測最大值以供參照。\n")

    txt += ("\n量測說明：各 GPU 階段後均強制同步，故所測為真實牆鐘時間；"
            "前若干影格之暖機已排除。ROMP 與其後各階段於兩個獨立環境量測，"
            "「各階段平均之和」代表**計算成本**，"
            "未含實際系統中行程間傳遞、系統排程與記憶體配置的開銷，"
            "故其所對應的更新率為推估值而非端到端實測值。\n")

    for m in meta:
        txt += f"\n- {m}"

    Path(args.out).write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n已寫出 {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["romp", "render", "endtoend"])
    p.add_argument("--report", nargs="+")
    p.add_argument("--video")
    p.add_argument("--motion")
    p.add_argument("--work", default="demo_work/A2_navy_coat")
    p.add_argument("--bind-pkl", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    p.add_argument("--smplx-model-dir", default="src/smpl_related/models")
    p.add_argument("--frames", type=int, default=250, help="有效量測影格數")
    p.add_argument("--warmup", type=int, default=20,
                   help="丟棄的暖機影格數。前幾影格含 CUDA 初始化與核心編譯，"
                        "耗時可達穩態數倍，必須排除")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.report:
        report(args)
    elif args.part == "romp":
        part_romp(args)
    elif args.part == "render":
        part_render(args)
    elif args.part == "endtoend":
        part_endtoend(args)
    else:
        p.error("需指定 --part 或 --report")


if __name__ == "__main__":
    main()