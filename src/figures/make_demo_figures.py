r"""
make_demo_figures.py —— 產生論文圖 4-15（即時驅動展示）與圖 4-16（文字換裝展示）

圖 4-15　即時姿態驅動展示
    上排為輸入影片的影格，下排為對應的著衣網格驅動結果。
    兩排取自**同一組影格**，故可直接比對姿態是否跟隨。

圖 4-16　文字換裝展示
    同一體型、同一姿態下，三套由不同文字提示生成的服裝並列。
    姿態固定是必要的——若各套服裝擺不同姿勢，讀者無從分辨差異來自
    服裝本身或姿態。

相機設定
========
本腳本採**正交投影**並依網格包圍盒自動取景，而非沿用 ROMP 的弱透視相機。
理由是這兩張圖的用途為「展示模型」而非「與真人對位」——後者已由圖 4-14
（真人影像合成）負責。正交投影可使各格的人體大小一致，便於比對。

圖 4-15 的各格**共用同一取景**（依全序列的包圍盒聯集計算），否則人體會
隨動作幅度忽大忽小，看起來像相機在推拉而非人在動。若想改回逐格取景，
加上 --per-frame-fit。

用法
====
  # 兩張圖一起產生
  python make_demo_figures.py --out figures

  # 只做其中一張
  python make_demo_figures.py --fig 15 --motion shoulder --video-index 1 --out figures
  python make_demo_figures.py --fig 16 --out figures

  # 看看有哪些資產可用
  python make_demo_figures.py --list-assets

環境：chupa（需 torch、nvdiffrast）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from drive_avatar import (load_bind, build_smplx, global_transforms,
                              motion_to_full_pose)
except ImportError as e:
    sys.exit(f"[錯誤] 需與 drive_avatar.py 置於同一目錄：{e}")

# 圖 4-16 所用的三套服裝。同一體型、不同文字提示。
OUTFITS = [
    # 標籤刻意寫 "red dress" 而非提示詞原文 "floor-length red dress"：
    # 生成結果為及膝長度，長度屬性未達成（見 4.2.5 節）。標籤若照抄提示詞
    # 會與圖面不符。
    ("demo_work/A1_red_dress_v2",  "red dress"),
    ("demo_work/A2_navy_coat_q50",  "navy coat + khaki pants"),
    ("demo_work/A3_green_hood_q50", "green hoodie"),
]

CANVAS = 512          # 每一格的邊長（像素）
N_COLS = 5            # 圖 4-15 取樣的影格數


def list_assets():
    """列出所有具備完整驅動所需檔案的資產目錄。"""
    roots = [Path("demo_work"), Path("work"), Path(".")]
    seen = set()
    print("可用的資產（含 geometry.npz + skin_weights.npz + texture.png）：\n")
    found = False
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("geometry.npz")):
            d = p.parent
            if d in seen:
                continue
            seen.add(d)
            has_w = (d / "skin_weights.npz").exists()
            has_t = (d / "texture.png").exists()
            if has_w and has_t:
                print(f"  ✓ {d}")
                found = True
            else:
                miss = []
                if not has_w:
                    miss.append("skin_weights.npz")
                if not has_t:
                    miss.append("texture.png")
                print(f"  ✗ {d}　缺 {', '.join(miss)}")
    if not found:
        print("\n（沒有完整的資產。skin_weights.npz 由 "
              "drive_avatar.py --stage weights 產生）")


def load_asset(work):
    """載入資產的幾何、UV、貼圖與蒙皮權重。"""
    import cv2
    work = Path(work)
    for f in ("geometry.npz", "skin_weights.npz", "texture.png"):
        if not (work / f).exists():
            sys.exit(f"[錯誤] {work} 缺少 {f}。"
                     f"可執行 python make_demo_figures.py --list-assets 查看可用資產")
    g = np.load(work / "geometry.npz")
    sw = np.load(work / "skin_weights.npz")
    tex = cv2.cvtColor(cv2.imread(str(work / "texture.png")),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return {
        "verts": g["verts"].astype(np.float64),
        "faces": g["faces"].astype(np.int32),
        "uv": g["uv"].astype(np.float32),
        "sj": sw["joints"].astype(np.int64),
        "sw": sw["weights"].astype(np.float64),
        "tex": tex,
    }


def drive(asset, bind, J_rest, parents, full_pose, G_bind_inv):
    """以指定姿態驅動資產，回傳世界座標下的頂點。"""
    G = global_transforms(J_rest, full_pose, parents)
    T = np.einsum("jab,jbc->jac", G, G_bind_inv)

    # 頂點先由世界座標還原至模型座標，蒙皮後再變換回模型座標系下的結果——
    # 關節變換係於模型座標系計算，不可混入資產的縮放與平移（3.3.1 節）。
    # 此處與 drive_avatar.py stage_export 的處理一致。
    v_model = (asset["verts"] - bind["transl"].reshape(1, 3)) / bind["scale"]
    vh = np.concatenate([v_model, np.ones((len(v_model), 1))], axis=1)
    M = np.einsum("nk,nkab->nab", asset["sw"], T[asset["sj"]])
    return np.einsum("nab,nb->na", M, vh)[:, :3]


def make_view(all_verts, margin=0.06):
    """由多組頂點計算共用的取景參數（中心與半寬）。"""
    xs = np.concatenate([v[:, 0] for v in all_verts])
    ys = np.concatenate([v[:, 1] for v in all_verts])
    cx = (xs.max() + xs.min()) / 2
    cy = (ys.max() + ys.min()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + margin)
    return float(cx), float(cy), float(half)


def vertex_normals(verts, faces):
    """由三角面法向量累加得到頂點法向量。

    面積加權為自然的結果——外積的長度即為兩倍三角形面積，故大面自動
    取得較高權重，不需另行加權。
    """
    n = np.zeros(verts.shape, np.float64)
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, faces[:, i], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def render(verts, asset, view, ctx, size=CANVAS, flip_y=True, back_view=False,
           mode="texture"):
    r"""正交投影渲染單一格。view 為 make_view 回傳的共用取景參數。

    mode="texture" 輸出貼圖後的彩現；mode="normal" 輸出法向量著色，
    用以呈現**未經貼圖的幾何**。後者是圖 4-16 上列的內容——貼圖會依
    人體部位語意區域上色，使裙襬以下裸露的小腿被塗成服裝顏色，單看
    貼圖後的結果無法分辨「及膝裙」與「長褲」。法向量著色不受此影響。
    """
    import torch
    import nvdiffrast.torch as dr

    v = verts.astype(np.float32)
    cx, cy, half = view

    clip = np.ones((len(v), 4), np.float32)
    clip[:, 0] = (v[:, 0] - cx) / half
    clip[:, 1] = (v[:, 1] - cy) / half
    # 深度符號：nvdiffrast 保留 NDC z 較小者（較近）。SMPL-X 的人體朝向 +Z，
    # 故正面頂點的世界 z 較大；若直接映射，正面會被判為較遠而遭背面遮蔽，
    # 渲染結果將是後腦杓。此處取負號，使 +Z 側朝向觀察者。
    zc = v[:, 2] - v[:, 2].mean()
    sign = 1.0 if back_view else -1.0
    clip[:, 2] = np.clip(sign * zc / (half * 4), -0.9, 0.9)

    dev = "cuda"
    clip_t = torch.tensor(clip, device=dev).unsqueeze(0)
    faces_t = torch.tensor(asset["faces"], dtype=torch.int32, device=dev)

    rast, _ = dr.rasterize(ctx, clip_t, faces_t, resolution=[size, size])
    vis = (rast[..., 3:4] > 0).float()

    if mode == "normal":
        nrm = vertex_normals(verts.astype(np.float64),
                             asset["faces"]).astype(np.float32)
        nrm_t = torch.tensor(nrm, device=dev).unsqueeze(0)
        ni, _ = dr.interpolate(nrm_t, rast, faces_t)
        # 內插後長度不再為 1，須重新正規化，否則面中央會偏暗
        ni = torch.nn.functional.normalize(ni, dim=-1)
        col = ni * 0.5 + 0.5
    else:
        uv = asset["uv"].copy()
        uv[:, 1] = 1.0 - uv[:, 1]    # nvdiffrast 的貼圖取樣需翻轉 v 軸（附錄 A.2）
        uv_t = torch.tensor(uv, device=dev).unsqueeze(0)
        tex_t = torch.tensor(asset["tex"], device=dev).unsqueeze(0)
        uvi, _ = dr.interpolate(uv_t, rast, faces_t)
        col = dr.texture(tex_t, uvi, filter_mode="linear")

    rgb = (col * vis)[0].detach().cpu().numpy()
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if flip_y:
        img = np.flipud(img)
    return np.ascontiguousarray(img)


def label_strip(imgs, texts, height=34):
    """在每格下方加上文字說明（僅 ASCII，OpenCV 內建字型不支援中文）。"""
    import cv2
    out = []
    for im, t in zip(imgs, texts):
        band = np.full((height, im.shape[1], 3), 255, np.uint8)
        cv2.putText(band, t, (8, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        out.append(np.vstack([im, band]))
    return np.hstack(out)


# ============================================================
# 圖 4-15：即時驅動展示
# ============================================================

def fig_driving(args):
    import cv2
    import nvdiffrast.torch as dr

    pose_p = Path(args.temporal) / f"pose_{args.motion}.npz"
    if not pose_p.exists():
        avail = sorted(Path(args.temporal).glob("pose_*.npz")) \
            if Path(args.temporal).is_dir() else []
        msg = "；已有：" + ", ".join(p.stem[5:] for p in avail) if avail else ""
        sys.exit(f"[錯誤] 找不到 {pose_p}{msg}")
    poses = np.load(pose_p)["poses"]

    vp = Path(os.path.expanduser(args.video_dir)) / f"{args.video_index}.mp4"
    if not vp.exists():
        sys.exit(f"[錯誤] 找不到影片 {vp}")

    asset = load_asset(args.work)
    bind = load_bind(args.bind)
    J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(global_transforms(J_rest, bind["full_pose"], parents))
    full = motion_to_full_pose(poses, bind["full_pose"])

    # 於序列中均勻取樣。避開頭尾各 10%，因該處常為預備與收勢動作，
    # 姿態變化小、不具展示價值。
    lo, hi = int(0.1 * len(full)), int(0.9 * len(full))
    idx = np.linspace(lo, hi - 1, args.cols).astype(int)
    print(f"[INFO] {args.motion}：{len(full)} 影格，取樣 {list(map(int, idx))}")

    driven = [drive(asset, bind, J_rest, parents, full[int(i)], G_bind_inv)
              for i in idx]
    # 各格共用同一取景，人體大小才不會隨動作幅度變化
    view = make_view(driven) if not args.per_frame_fit else None

    cap = cv2.VideoCapture(str(vp))
    frames, k, want = {}, 0, set(int(i) for i in idx)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            frames[k] = fr
        k += 1
    cap.release()
    if len(frames) < len(want):
        print(f"  [警告] 影片僅 {k} 影格，姿態序列有 {len(full)} 影格，"
              f"缺 {len(want) - len(frames)} 格")

    ctx = dr.RasterizeCudaContext()
    top, bottom = [], []
    for i, v in zip(idx, driven):
        fr = frames.get(int(i))
        if fr is None:
            fr = np.zeros((CANVAS, CANVAS, 3), np.uint8)
        else:
            h, w = fr.shape[:2]
            s = min(h, w)
            fr = fr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        top.append(cv2.resize(fr, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA))

        vw = view if view is not None else make_view([v])
        bottom.append(cv2.cvtColor(render(v, asset, vw, ctx,
                                          flip_y=not args.no_flip_y,
                                          back_view=args.back_view),
                                   cv2.COLOR_RGB2BGR))

    labels = [f"frame {int(i)}" for i in idx]
    strip = np.vstack([
        label_strip(top, ["input video"] + [""] * (len(top) - 1)),
        label_strip(bottom, labels),
    ])
    out = Path(args.out) / "fig_4-15_driving.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)
    print(f"[完成] {out}")


# ============================================================
# 圖 4-16：文字換裝展示
# ============================================================

def fig_outfits(args):
    import cv2
    import nvdiffrast.torch as dr

    bind = load_bind(args.bind)
    J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(global_transforms(J_rest, bind["full_pose"], parents))

    # 三套服裝**共用同一姿態**。若各自擺不同姿勢，讀者無從分辨
    # 差異來自服裝或姿態。
    if args.pose_from:
        poses = np.load(Path(args.temporal) / f"pose_{args.pose_from}.npz")["poses"]
        full = motion_to_full_pose(poses, bind["full_pose"])
        pose = full[len(full) // 2]
        print(f"[INFO] 三套服裝共用 {args.pose_from} 之第 {len(full) // 2} 影格姿態")
    else:
        pose = bind["full_pose"]
        print("[INFO] 三套服裝共用綁定姿勢（A-pose）")

    assets, driven, texts = [], [], []
    for work, label in OUTFITS:
        p = Path(work)
        if not (p / "geometry.npz").exists() or not (p / "skin_weights.npz").exists():
            print(f"  [略過] {work}")
            continue
        a = load_asset(work)
        assets.append(a)
        driven.append(drive(a, bind, J_rest, parents, pose, G_bind_inv))
        texts.append(label)

    if not driven:
        sys.exit("[錯誤] 沒有可用的服裝資產。"
                 "執行 python make_demo_figures.py --list-assets 查看，"
                 "或修改本檔開頭的 OUTFITS")

    # 三格共用取景，各套服裝的體型比例才可直接比對
    view = make_view(driven)
    ctx = dr.RasterizeCudaContext()
    def row(mode):
        return [cv2.cvtColor(render(v, a, view, ctx, mode=mode,
                                    flip_y=not args.no_flip_y,
                                    back_view=args.back_view),
                             cv2.COLOR_RGB2BGR)
                for v, a in zip(driven, assets)]

    imgs = row("texture")

    out = Path(args.out) / "fig_4-16_outfits.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.single_row:
        cv2.imwrite(str(out), label_strip(imgs, texts))
    else:
        # 上列幾何、下列貼圖，兩列共用同一取景與同一姿態，故可逐格對照：
        # 讀者先由上列確認幾何的服裝類型正確，再由下列看出貼圖的區域界定
        # 誤差（裙襬以下的小腿被塗上服裝顏色）。單看下列無法區分「及膝裙」
        # 與「長褲」，此即本圖需要兩列的理由。
        strip = np.vstack([
            label_strip(row("normal"),
                        ["geometry (normals)"] + [""] * (len(imgs) - 1)),
            label_strip(imgs, texts),
        ])
        cv2.imwrite(str(out), strip)

    print(f"[完成] {out}（{len(imgs)} 套服裝，"
          f"{'單列' if args.single_row else '幾何／貼圖兩列'}）")


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-15 與圖 4-16")
    ap.add_argument("--fig", choices=["15", "16", "both"], default="both")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--work", default="demo_work/A2_navy_coat_q50",
                    help="圖 4-15 所用的資產")
    ap.add_argument("--bind", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    ap.add_argument("--model-dir", default="src/smpl_related/models")
    ap.add_argument("--temporal", default="temporal_eval")
    ap.add_argument("--motion", default="shoulder", help="圖 4-15 所用的動作段")
    ap.add_argument("--video-index", default="1",
                    help="對應的影片編號（1=shoulder、2=elbow、3=run、"
                         "4=legraise、5=twist、6=stress_arms）")
    ap.add_argument("--video-dir", default="~/projects/chupa/motion_testVideo")
    ap.add_argument("--cols", type=int, default=N_COLS)
    ap.add_argument("--pose-from",
                    help="圖 4-16 借用哪一段動作的姿態；省略則用 A-pose")
    ap.add_argument("--per-frame-fit", action="store_true",
                    help="圖 4-15 改為逐格各自取景（預設共用取景）")
    ap.add_argument("--no-flip-y", action="store_true",
                    help="若輸出上下顛倒，加上此旗標")
    ap.add_argument("--back-view", action="store_true",
                    help="改渲染背面（預設為正面）")
    ap.add_argument("--single-row", action="store_true",
                    help="圖 4-16 只輸出貼圖列（預設為幾何／貼圖兩列）")
    ap.add_argument("--list-assets", action="store_true")
    args = ap.parse_args()

    if args.list_assets:
        list_assets()
        return
    if args.fig in ("15", "both"):
        fig_driving(args)
    if args.fig in ("16", "both"):
        fig_outfits(args)


if __name__ == "__main__":
    main()
