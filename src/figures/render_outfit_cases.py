"""
render_outfit_cases.py

把九組「體型 × 顏色 × 服裝」的貼圖資產（chupa_textured.glb）
排成 3x3 彩色案例圖，同一比例尺、三列各自對齊地面線。

在 chupa 環境、chupa 專案目錄執行：

    python render_outfit_cases.py
    python render_outfit_cases.py --rot-y 180      # 若出來是背面
    python render_outfit_cases.py --out fig.png

輸出：outfit_cases_figure.png
"""
import argparse
import csv
import os
import sys
import textwrap

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

INDEX_CSV = "texture_cases_index.csv"
DPI = 200
LIGHT_DIR = np.array([-0.35, 0.45, 0.85])
AMBIENT, DIFFUSE = 0.45, 0.55          # 貼圖已有顏色，陰影放輕一點
FALLBACK_COLOR = np.array([0.74, 0.76, 0.80])

_CJK = ["Noto Sans CJK TC", "Noto Sans CJK JP", "Microsoft JhengHei",
        "PingFang TC", "WenQuanYi Zen Hei", "Source Han Sans TW"]
_avail = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_font = next((f for f in _CJK if f in _avail), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
    plt.rcParams["axes.unicode_minus"] = False


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def load_textured(path):
    """讀 glb，回傳 (頂點, 面, 頂點法向量, 每面顏色)。沒有貼圖則回傳灰色。"""
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"{path} 不含任何網格")
        mesh = max(loaded.geometry.values(), key=lambda g: len(g.faces))
    else:
        mesh = loaded

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)

    face_rgb = None
    try:
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        tex = mesh.visual.material.baseColorTexture
        if uv is not None and tex is not None:
            img = np.asarray(tex.convert("RGB"), dtype=np.float64) / 255.0
            H, W = img.shape[:2]
            c = uv[f].mean(axis=1)                       # 每面的 UV 重心
            px = np.clip((c[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
            py = np.clip((1.0 - c[:, 1] % 1.0) * (H - 1), 0, H - 1).astype(int)
            face_rgb = img[py, px]
    except Exception as e:
        print(f"  [警告] {os.path.basename(path)} 取貼圖失敗（{e}），改用灰色")

    if face_rgb is None:
        face_rgb = np.tile(FALLBACK_COLOR, (len(f), 1))
    return v, f, vn, face_rgb


def shaded(v, f, vn, face_rgb, deg):
    R = rot_y(deg)
    vr, nr = v @ R.T, vn @ R.T
    tri = vr[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = fn[:, 2] > 0
    tri, ff, rgb = tri[front], f[front], face_rgb[front]

    order = np.argsort(tri[:, :, 2].mean(axis=1))
    tri, ff, rgb = tri[order], ff[order], rgb[order]

    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    sh = np.clip(nr[ff] @ light, 0, 1).mean(axis=1)
    colors = np.clip(rgb * (AMBIENT + DIFFUSE * sh)[:, None], 0, 1)
    return tri[:, :, :2], colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-y", type=float, default=0.0, help="繞垂直軸旋轉角度")
    ap.add_argument("--out", default="outfit_cases_figure.png")
    ap.add_argument("--gap", type=float, default=1.18,
                    help="左右間距倍率，標籤重疊時調大")
    args = ap.parse_args()

    if not os.path.exists(INDEX_CSV):
        sys.exit(f"[錯誤] 找不到 {INDEX_CSV}")
    rows = sorted(csv.DictReader(open(INDEX_CSV, encoding="utf-8-sig")),
                  key=lambda r: int(r["no"]))

    items = []
    for r in rows:
        glb = os.path.join(r["work"], "chupa_textured.glb")
        if not os.path.exists(glb):
            print(f"  [略過] 第 {r['no']} 組找不到 {glb}")
            continue
        v, f, vn, rgb = load_textured(glb)
        v = v - np.array([0, v[:, 1].min(), 0])
        hgt = v[:, 1].max()
        torso = (v[:, 1] > 0.45 * hgt) & (v[:, 1] < 0.65 * hgt)
        if not torso.any():
            torso = np.ones(len(v), dtype=bool)
        v[:, 0] -= np.median(v[torso, 0])
        v[:, 2] -= np.median(v[torso, 2])

        parts = r["subject"].split("_")          # rp_lib_<h>_<w>_001
        h, w = (parts[2], parts[3]) if len(parts) >= 4 else ("?", "?")
        items.append(dict(v=v, f=f, vn=vn, rgb=rgb, top=hgt, h=h, w=w,
                          no=r["no"], prompt=r["prompt"]))
        print(f"  [載入] 第 {r['no']} 組 {h}cm/{w}kg  {len(v)} 頂點")

    if not items:
        sys.exit("[錯誤] 沒有可用的 glb")

    max_top = max(it["top"] for it in items)
    half_w = max(np.abs(it["v"][:, [0, 2]]).max() for it in items) * 1.05
    n_col = 3
    n_row = int(np.ceil(len(items) / n_col))

    fig, axes = plt.subplots(n_row, 1, figsize=(4.0 * n_col, 5.2 * n_row), dpi=DPI)
    axes = np.atleast_1d(axes)
    spacing = 2 * half_w * args.gap

    for ri, ax in enumerate(axes):
        row_items = items[ri * n_col:(ri + 1) * n_col]
        for ci, it in enumerate(row_items):
            polys, colors = shaded(it["v"], it["f"], it["vn"], it["rgb"], args.rot_y)
            ax.add_collection(PolyCollection(
                polys + np.array([ci * spacing, 0.0]),
                facecolors=colors, edgecolors=colors,
                linewidths=0.12, antialiaseds=True))
            cx = ci * spacing
            # 服裝描述：去掉開頭的 "a woman wearing "
            desc = it["prompt"].replace("a woman wearing ", "").strip()
            desc = "\n".join(textwrap.wrap(desc, width=26)) or desc
            ax.text(cx, -0.048 * max_top, f"({it['no']}) {it['h']} cm / {it['w']} kg",
                    fontsize=11, ha="center", va="top", fontweight="bold")
            ax.text(cx, -0.092 * max_top, desc, fontsize=8.5, ha="center",
                    va="top", color="#444444", linespacing=1.4)

        ax.plot([-half_w, (n_col - 1) * spacing + half_w], [0, 0],
                color="#333333", lw=1.0)
        ax.set_xlim(-half_w, (n_col - 1) * spacing + half_w)
        ax.set_ylim(-0.17 * max_top, 1.06 * max_top)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.06)
    fig.savefig(args.out, dpi=DPI, facecolor="white")
    print(f"\n[完成] 已輸出 {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
