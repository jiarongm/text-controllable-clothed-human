r"""
make_turntable.py —— 將著衣資產錄製成繞軸轉一圈的影片

用於展示單一資產的完整外觀（正面、側面、背面皆可見），適合口試簡報，
亦可自輸出中擷取數個角度排成圖。

兩項必須固定的設定
==================
1. **取景固定。** 若逐影格依當下的包圍盒取景，人體會隨旋轉忽大忽小，
   看起來像相機在推拉而非模型在轉。本腳本先掃過整圈求出最大半徑，
   再以該值固定視野。
2. **旋轉軸過身體中心。** 否則模型會繞著畫面外的某點公轉。本腳本以
   包圍盒中心的 x、z 為軸。

貼圖的選擇
==========
資產目錄下可能同時存在 `texture.png`（未修補的原始圖集）與
`chupa_textured_texture.png`（修補後）。本腳本預設取後者；若僅有前者，
渲染會呈椒鹽狀雜點，屆時會明確警告。

用法
====
  conda activate chupa
  cd ~/projects/chupa

  python make_turntable.py --work demo_work/U_user_155_80 \
      --out overlay_check/turntable_user_155_80.mp4

  # 幾何版（法向量著色，不貼圖）——看體型輪廓時較清楚
  python make_turntable.py --work demo_work/U_user_155_80 --shading normal \
      --out overlay_check/turntable_geom.mp4

  # 同時輸出四個角度的靜態圖（0°/90°/180°/270°）
  python make_turntable.py --work demo_work/U_user_155_80 --contact-sheet

環境：chupa（需 torch、nvdiffrast、opencv）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

TEX_CANDIDATES = [
    "final/chupa_textured_texture.png",
    "chupa_textured_texture.png",
    "texture.png",
]
RAW_TEX = "texture.png"


def find_texture(work):
    for rel in TEX_CANDIDATES:
        p = work / rel
        if p.exists():
            return p, (rel == RAW_TEX)
    return None, False


def vertex_normals(v, f):
    n = np.zeros_like(v, dtype=np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def rotate_y(v, center, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    p = v - center
    x = p[:, 0] * c + p[:, 2] * s
    z = -p[:, 0] * s + p[:, 2] * c
    out = p.copy()
    out[:, 0], out[:, 2] = x, z
    return out + center


def render(verts, asset, frame, size, mode, flip_y=True):
    import torch
    import nvdiffrast.torch as dr

    v = verts.astype(np.float32)
    cx, cy, half = frame
    clip = np.ones((len(v), 4), np.float32)
    clip[:, 0] = (v[:, 0] - cx) / half
    clip[:, 1] = (v[:, 1] - cy) / half
    # 深度取負：SMPL-X 人體朝 +Z，而 nvdiffrast 保留 NDC z 較小者。
    # 旋轉後此關係仍成立，因為旋轉作用於頂點本身而非相機。
    clip[:, 2] = np.clip(-(v[:, 2] - v[:, 2].mean()) / (half * 4), -0.9, 0.9)

    dev = "cuda"
    ctx = asset["_ctx"]
    clip_t = torch.tensor(clip, device=dev).unsqueeze(0)
    faces_t = asset["_faces_t"]
    rast, _ = dr.rasterize(ctx, clip_t, faces_t, resolution=[size, size])
    vis = (rast[..., 3:4] > 0).float()

    if mode == "texture":
        uvi, _ = dr.interpolate(asset["_uv_t"], rast, faces_t)
        col = dr.texture(asset["_tex_t"], uvi, filter_mode="linear")
    else:
        nrm = vertex_normals(verts, asset["faces"].astype(np.int64))
        nt = torch.tensor(nrm.astype(np.float32), device=dev).unsqueeze(0)
        ni, _ = dr.interpolate(nt, rast, faces_t)
        ni = torch.nn.functional.normalize(ni, dim=-1)
        col = (0.30 + 0.70 * torch.clamp(-ni[..., 2:3], 0, 1)).repeat(1, 1, 1, 3)

    rgb = (col * vis + (1.0 - vis))[0].detach().cpu().numpy()
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if flip_y:
        img = np.flipud(img)
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser(description="錄製資產的轉台影片")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", type=int, default=120, help="一圈的影格數")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--shading", choices=["texture", "normal"],
                    default="texture")
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--contact-sheet", action="store_true",
                    help="另輸出 0/90/180/270 度的四格靜態圖")
    ap.add_argument("--no-flip-y", action="store_true")
    args = ap.parse_args()

    import cv2
    import torch
    import nvdiffrast.torch as dr

    work = Path(args.work)
    if not work.is_absolute():
        work = ROOT / work
    if not (work / "geometry.npz").exists():
        sys.exit(f"[錯誤] {work} 缺少 geometry.npz")

    g = np.load(work / "geometry.npz")
    asset = {"verts": g["verts"].astype(np.float64),
             "faces": g["faces"].astype(np.int32),
             "uv": g["uv"].astype(np.float32)}

    tex, raw = find_texture(work)
    if args.shading == "texture":
        if tex is None:
            sys.exit(f"[錯誤] {work} 下找不到貼圖；可改用 --shading normal")
        if raw:
            print(f"[警告] 僅有未修補的原始圖集 {RAW_TEX}，"
                  "渲染會呈椒鹽狀雜點。\n       建議改用 --shading normal，"
                  "或確認是否漏跑 export 步驟。")
        print(f"[INFO] 貼圖：{tex.relative_to(work)}")
        t = cv2.cvtColor(cv2.imread(str(tex)),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        asset["_tex_t"] = torch.tensor(t, device="cuda").unsqueeze(0)
        uv = asset["uv"].copy()
        uv[:, 1] = 1.0 - uv[:, 1]            # 附錄 A.2
        asset["_uv_t"] = torch.tensor(uv, device="cuda").unsqueeze(0)

    asset["_ctx"] = dr.RasterizeCudaContext()
    asset["_faces_t"] = torch.tensor(asset["faces"], dtype=torch.int32,
                                     device="cuda")

    v = asset["verts"]
    center = np.array([(v[:, 0].max() + v[:, 0].min()) / 2, 0.0,
                       (v[:, 2].max() + v[:, 2].min()) / 2])
    center[1] = (v[:, 1].max() + v[:, 1].min()) / 2

    # 固定取景：取整圈中最大的水平半徑，避免模型隨旋轉忽大忽小
    r_xz = float(np.sqrt(((v[:, [0, 2]] - center[[0, 2]]) ** 2).sum(1)).max())
    half_y = float(v[:, 1].ptp() / 2)
    half = max(r_xz, half_y) * (1.0 + args.margin)
    frame = (center[0], center[1], half)
    print(f"[INFO] 旋轉中心 {center.round(4)}，固定半視野 {half:.4f}"
          f"（水平半徑 {r_xz:.4f}、垂直半高 {half_y:.4f}）")

    out = Path(args.out) if args.out else \
        ROOT / f"turntable_{work.name}_{args.shading}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (args.size, args.size))
    if not vw.isOpened():
        sys.exit(f"[錯誤] 無法建立影片 {out}")

    sheet = {}
    for i in range(args.frames):
        deg = 360.0 * i / args.frames
        rv = rotate_y(v, center, deg)
        img = render(rv, asset, frame, args.size, args.shading,
                     flip_y=not args.no_flip_y)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for d in (0, 90, 180, 270):
            if abs(deg - d) < 360.0 / args.frames / 2:
                sheet[d] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if (i + 1) % 30 == 0 or i + 1 == args.frames:
            print(f"  {i + 1}/{args.frames}")
    vw.release()
    print(f"\n[完成] {out}（{args.frames} 影格 @ {args.fps} fps，"
          f"{args.frames / args.fps:.1f} 秒）")

    if args.contact_sheet:
        have = [d for d in (0, 90, 180, 270) if d in sheet]
        if have:
            band = 34
            tiles = []
            for d in have:
                im = sheet[d]
                b = np.full((band, im.shape[1], 3), 255, np.uint8)
                cv2.putText(b, f"{d} deg", (8, band - 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                            cv2.LINE_AA)
                tiles.append(np.vstack([im, b]))
            sp = out.with_name(out.stem + "_sheet.png")
            cv2.imwrite(str(sp), np.hstack(tiles))
            print(f"[完成] {sp}（{len(have)} 個角度，共用取景）")


if __name__ == "__main__":
    main()
