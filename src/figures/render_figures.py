r"""
render_figures.py

把消融實驗的 .glb 以**完全相同的相機、光照與構圖**渲染，並組成論文用的對照圖。

為什麼要另外寫腳本而非手動截圖：三張對照圖中，`baseline` 這一組會重複出現於
圖 4-9(c)、圖 4-10(b)、圖 4-11(a)。若各圖分別手動截圖，視角或縮放的細微差異
會讓讀者誤以為那是三個不同的結果。本腳本對所有組別套用同一組相機參數，
並讓重複出現的組別共用同一張渲染輸出，從源頭排除此問題。

渲染採 Blender 的 Workbench 引擎、平面光照（flat）、材質色彩模式。
此設定直接呈現貼圖的原始色彩，不引入光照造成的明暗變化——對「比較貼圖」
這個目的而言，任何額外的打光都是干擾項。若需帶陰影的展示圖，改用 --engine eevee。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  cd ~/projects/chupa

  # 先確認 Blender 可執行
  blender --version

  python render_figures.py --runs thesis_figures/_runs \
      --out thesis_figures/rendered

  # 同時輸出背面（語意約束的漂移在背面最明顯）
  python render_figures.py --runs thesis_figures/_runs \
      --out thesis_figures/rendered --views front back

  # Blender 不在 PATH 時
  python render_figures.py --blender /path/to/blender ...

輸出：
  thesis_figures/rendered/
  ├── panels/                      各組別的個別渲染（含 alpha）
  │   ├── baseline_front.png
  │   ├── no_semantic_front.png
  │   └── ...
  ├── fig4-9_semantic_front.png    三組並排，含 (a)(b)(c) 標記
  ├── fig4-10_accumulate_front.png
  ├── fig4-11_recolor_front.png
  └── README.md                    各圖對應之組別說明

個別渲染（panels/）帶 alpha 通道，若你偏好在 Word 或 LaTeX 中自行排版並加上
子標題，直接使用它們即可；並排圖僅為方便快速檢視與投影。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2


# 三張圖各由哪些組別構成。baseline 重複出現於三張圖中，
# 但只會被渲染一次、三處共用同一檔案。
FIGURES = {
    "fig4-9_semantic": [
        ("no_semantic", "(a) no semantic constraint"),
        ("no_smooth",   "(b) constraint, no smoothing"),
        ("baseline",    "(c) constraint + smoothing"),
    ],
    "fig4-10_accumulate": [
        ("accum_full", "(a) accumulate full image"),
        ("baseline",   "(b) accumulate new pixels only"),
    ],
    "fig4-11_recolor": [
        ("baseline",   "(a) without hue correction"),
        ("recolor_on", "(b) with hue correction"),
    ],
}


BLENDER_SCRIPT = r'''
import bpy, sys, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
glb_path, out_path, view, engine, res = argv[0], argv[1], argv[2], argv[3], int(argv[4])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if not meshes:
    raise RuntimeError("glb 裡沒有 mesh")

# 世界座標下的包圍盒。五組資產的幾何完全相同（僅貼圖不同），
# 故此處算出的構圖對各組一致；即使不同也會因同一算法而可比。
lo = Vector((1e9, 1e9, 1e9))
hi = Vector((-1e9, -1e9, -1e9))
for o in meshes:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        lo = Vector((min(lo[i], w[i]) for i in range(3)))
        hi = Vector((max(hi[i], w[i]) for i in range(3)))
center = (lo + hi) / 2
size = hi - lo

# glTF 為 Y-up，Blender 匯入時已轉為 Z-up，故此處以 Z 為高度、Y 為前後
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = max(size.x, size.z) * 1.15
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)

dist = max(size.y, 1.0) * 5
if view == "back":
    cam.location = (center.x, center.y + dist, center.z)
    cam.rotation_euler = (math.radians(90), 0, math.radians(180))
else:
    cam.location = (center.x, center.y - dist, center.z)
    cam.rotation_euler = (math.radians(90), 0, 0)
bpy.context.scene.camera = cam

sc = bpy.context.scene
sc.render.resolution_x = res
sc.render.resolution_y = res
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.filepath = out_path

if engine == "eevee":
    # 版本間名稱不同，逐一嘗試
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = name
            break
        except TypeError:
            continue
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.5
    sc.world = world
else:
    # Workbench + 平面光照：直接呈現貼圖原色，不引入打光造成的明暗
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "TEXTURE"

bpy.ops.render.render(write_still=True)
'''


def find_blender(explicit=None):
    if explicit:
        return explicit
    for cand in ("blender", "/usr/bin/blender", "/snap/bin/blender"):
        if shutil.which(cand) or os.path.exists(cand):
            return cand
    return None


def render_one(blender, script_path, glb, out_png, view, engine, res):
    cmd = [blender, "--background", "--python", str(script_path), "--",
           str(glb), str(out_png), view, engine, str(res)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_png).exists():
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"渲染失敗: {glb} ({view})")


def on_white(img):
    """把帶 alpha 的渲染合成到白底，供論文使用。"""
    if img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32)
        return (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


def crop_to_content(imgs, pad=0.04):
    """依所有影像的聯集內容裁切。

    使用聯集而非各自裁切，是為了讓並排的各組維持相同尺度——
    若各自裁切，人物在畫面中的大小會因輪廓些微差異而不一致。
    """
    box = None
    for im in imgs:
        if im.shape[2] < 4:
            return imgs
        ys, xs = np.where(im[..., 3] > 8)
        if len(ys) == 0:
            continue
        b = [xs.min(), ys.min(), xs.max(), ys.max()]
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3])]
    if box is None:
        return imgs
    h, w = imgs[0].shape[:2]
    m = int(pad * max(box[2] - box[0], box[3] - box[1]))
    x0, y0 = max(0, box[0] - m), max(0, box[1] - m)
    x1, y1 = min(w, box[2] + m), min(h, box[3] + m)
    return [im[y0:y1, x0:x1] for im in imgs]


def compose(panel_paths, labels, out_path, label_h=52):
    imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in panel_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return False
    imgs = crop_to_content(imgs)

    cols = []
    for im, lab in zip(imgs, labels):
        im = on_white(im)
        band = np.full((label_h, im.shape[1], 3), 255, np.uint8)
        (tw, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        x = max(4, (im.shape[1] - tw) // 2)
        cv2.putText(band, lab, (x, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (30, 30, 30), 2, cv2.LINE_AA)
        cols.append(np.vstack([im, band]))

    gap = np.full((cols[0].shape[0], 24, 3), 255, np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, gap, c])
    cv2.imwrite(str(out_path), out)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="thesis_figures/_runs",
                   help="消融輸出所在資料夾（含各組 .glb）")
    p.add_argument("--out", default="thesis_figures/rendered")
    p.add_argument("--views", nargs="+", default=["front"],
                   choices=["front", "back"],
                   help="要渲染的視角。語意約束的漂移在背面較明顯，"
                        "建議一併輸出 back")
    p.add_argument("--engine", default="workbench",
                   choices=["workbench", "eevee"],
                   help="workbench=平面光照、直接呈現貼圖原色（比較貼圖用）；"
                        "eevee=帶陰影（展示用）")
    p.add_argument("--res", type=int, default=1200)
    p.add_argument("--blender", default=None)
    args = p.parse_args()

    blender = find_blender(args.blender)
    if not blender:
        print("[ERROR] 找不到 blender。請用 --blender 指定執行檔路徑。")
        sys.exit(1)

    runs = Path(args.runs)
    out = Path(args.out)
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)

    # 收集所有需要渲染的組別（去重 —— baseline 出現於三張圖中）
    keys = []
    for items in FIGURES.values():
        for k, _ in items:
            if k not in keys:
                keys.append(k)

    missing = [k for k in keys if not (runs / f"{k}.glb").exists()]
    if missing:
        print(f"[ERROR] 找不到這些組別的 glb：{', '.join(missing)}")
        print(f"        請確認 {runs} 底下有對應的 .glb 檔。")
        print(f"        若只有貼圖沒有 glb，代表消融執行時 trimesh 未安裝，")
        print(f"        請於 chupa_tex 環境 pip install trimesh 後重跑消融。")
        sys.exit(1)

    print(f"Blender : {blender}")
    print(f"引擎    : {args.engine}")
    print(f"視角    : {', '.join(args.views)}")
    print(f"組別    : {len(keys)} 個（baseline 於三張圖中共用同一渲染）\n")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(BLENDER_SCRIPT)
        script_path = f.name

    try:
        n = 0
        for view in args.views:
            for k in keys:
                dst = panels / f"{k}_{view}.png"
                print(f"  渲染 {k} ({view}) ...")
                render_one(blender, script_path, runs / f"{k}.glb",
                           dst, view, args.engine, args.res)
                n += 1
        print(f"\n完成 {n} 張個別渲染")

        made = []
        for view in args.views:
            for fig, items in FIGURES.items():
                paths = [panels / f"{k}_{view}.png" for k, _ in items]
                labels = [lab for _, lab in items]
                dst = out / f"{fig}_{view}.png"
                if compose(paths, labels, dst):
                    made.append(dst)
                    print(f"  對照圖 {dst.name}")
    finally:
        os.unlink(script_path)

    with open(out / "README.md", "w", encoding="utf-8") as f:
        f.write("# 論文對照圖\n\n")
        f.write(f"渲染設定：Blender {args.engine} 引擎，"
                f"{'平面光照、材質色彩模式' if args.engine == 'workbench' else '含環境光'}，"
                f"解析度 {args.res}×{args.res}，透明背景。\n\n")
        f.write("所有組別套用同一組相機參數與構圖；`baseline` 於三張圖中"
                "共用同一張渲染輸出，以免讀者誤判為不同結果。\n\n")
        for fig, items in FIGURES.items():
            f.write(f"## {fig}\n\n| 位置 | 組別 | 說明 |\n|---|---|---|\n")
            for k, lab in items:
                f.write(f"| {lab.split(')')[0]}) | `{k}` | {lab.split(') ')[1]} |\n")
            f.write("\n")
        f.write("## 排版建議\n\n")
        f.write("`panels/` 內為各組的個別渲染（帶 alpha 通道），"
                "若於文件中自行排版並加上子標題，建議直接使用這些檔案；"
                "並排圖僅供快速檢視與投影。\n")

    print(f"\n輸出：{out.resolve()}")
    print(f"說明：{out / 'README.md'}")


if __name__ == "__main__":
    main()
