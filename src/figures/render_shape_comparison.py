"""
render_shape_comparison.py

產生三個體型的並排對照圖（上排正面、下排 45 度），
同一比例尺、腳底對齊同一條地面線，並標示身高體重。

只需要 numpy、trimesh、matplotlib，不需要 GPU 或顯示器。
在 chupa 環境、chupa 專案目錄執行：

    python render_shape_comparison.py

輸出：shape_comparison_figure.png
"""
import glob
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# ---------------- 設定 ----------------
SHAPES = [
    ("rp_ext_155_75_001", 155, 75),
    ("rp_ext_165_60_001", 165, 60),
    ("rp_ext_180_58_001", 180, 58),
]
PROMPT_KEY = "a_woman_wearing_a_tank_top_and_leggings"
MESH_PATTERN = "results/renderpeople/*/{subject}/*{key}*/mesh_refine.obj"
VIEWS = [("正面", 0.0), ("45°", 45.0)]      # (列標題, 繞垂直軸旋轉角度)
OUT_PATH = "shape_comparison_figure.png"
DPI = 220
BODY_COLOR = np.array([0.74, 0.76, 0.80])
LIGHT_DIR = np.array([-0.35, 0.45, 0.85])   # 相機座標：左上前方打光
# --------------------------------------

# 中文字型：找得到就用，找不到就改用英文標籤
_CJK = ["Noto Sans CJK TC", "Noto Sans CJK JP", "Microsoft JhengHei",
        "PingFang TC", "WenQuanYi Zen Hei", "Source Han Sans TW"]
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_font = next((f for f in _CJK if f in _available), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font]
    plt.rcParams["axes.unicode_minus"] = False
else:
    VIEWS = [("Front", 0.0), ("45 deg", 45.0)]


def find_mesh(subject):
    cands = glob.glob(MESH_PATTERN.format(subject=subject, key=PROMPT_KEY))
    return max(cands, key=os.path.getmtime) if cands else None


def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def shaded_polys(verts, faces, vnormals, deg):
    """正交投影 + 背面剔除 + 由遠到近排序，回傳 (2D 三角形, 顏色)。"""
    R = rot_y(deg)
    v = verts @ R.T
    n = vnormals @ R.T

    tri = v[faces]                                   # (F,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = fn[:, 2] > 0                             # 朝向相機（+Z）
    tri, f = tri[front], faces[front]

    order = np.argsort(tri[:, :, 2].mean(axis=1))    # 遠的先畫
    tri, f = tri[order], f[order]

    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    shade = np.clip(n[f] @ light, 0, 1).mean(axis=1)  # 用頂點法向量平均，較平滑
    shade = 0.30 + 0.70 * shade
    colors = np.clip(shade[:, None] * BODY_COLOR[None, :], 0, 1)
    return tri[:, :, :2], colors


def main():
    meshes = []
    for subject, h, w in SHAPES:
        path = find_mesh(subject)
        if path is None:
            sys.exit(f"[錯誤] 找不到 {subject} 的 mesh_refine.obj")
        m = trimesh.load(path, force="mesh", process=False)
        v = np.asarray(m.vertices, dtype=np.float64)
        f = np.asarray(m.faces, dtype=np.int64)
        vn = np.asarray(m.vertex_normals, dtype=np.float64)
        # 腳底對齊 y=0；x、z 以軀幹（中段高度）為中心，避免手臂影響置中
        v = v - np.array([0, v[:, 1].min(), 0])
        hgt = v[:, 1].max()
        torso = (v[:, 1] > 0.45 * hgt) & (v[:, 1] < 0.65 * hgt)
        if not torso.any():
            torso = np.ones(len(v), dtype=bool)
        v[:, 0] -= np.median(v[torso, 0])
        v[:, 2] -= np.median(v[torso, 2])
        meshes.append(dict(v=v, f=f, vn=vn, h=h, w=w, top=hgt))
        print(f"[載入] {subject}: {len(v)} 頂點, 網格高度 {hgt:.3f}  ({path})")

    max_top = max(m["top"] for m in meshes)
    half_w = max(np.abs(m["v"][:, [0, 2]]).max() for m in meshes) * 1.05
    spacing = 2 * half_w
    n_col, n_row = len(meshes), len(VIEWS)

    fig, axes = plt.subplots(
        n_row, 1, figsize=(4.2 * n_col, 5.0 * n_row), dpi=DPI)
    axes = np.atleast_1d(axes)

    for ax, (view_name, deg) in zip(axes, VIEWS):
        for i, m in enumerate(meshes):
            polys, colors = shaded_polys(m["v"], m["f"], m["vn"], deg)
            polys = polys + np.array([i * spacing, 0.0])
            ax.add_collection(PolyCollection(
                polys, facecolors=colors, edgecolors=colors,
                linewidths=0.15, antialiaseds=True))

            cx = i * spacing
            # 身高參考線
            ax.plot([cx - half_w * 0.92, cx + half_w * 0.92],
                    [m["top"], m["top"]],
                    color="#c0392b", lw=0.8, ls=(0, (4, 3)))
            ax.text(cx + half_w * 0.92, m["top"] + 0.012 * max_top,
                    f"{m['h']} cm", color="#c0392b", fontsize=10,
                    ha="right", va="bottom")
            # 體型標籤
            bmi = m["w"] / (m["h"] / 100) ** 2
            ax.text(cx, -0.055 * max_top,
                    f"{m['h']} cm / {m['w']} kg",
                    fontsize=12, ha="center", va="top", fontweight="bold")
            ax.text(cx, -0.105 * max_top, f"BMI {bmi:.1f}",
                    fontsize=10, ha="center", va="top", color="#555555")

        # 地面線
        ax.plot([-half_w, (n_col - 1) * spacing + half_w], [0, 0],
                color="#333333", lw=1.2)
        ax.text(-half_w, max_top * 1.04, view_name, fontsize=13,
                ha="left", va="bottom", color="#333333")

        ax.set_xlim(-half_w, (n_col - 1) * spacing + half_w)
        ax.set_ylim(-0.16 * max_top, 1.10 * max_top)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.04)
    fig.savefig(OUT_PATH, dpi=DPI, facecolor="white")
    print(f"\n[完成] 已輸出 {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
