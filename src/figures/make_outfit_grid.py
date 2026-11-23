r"""
make_outfit_grid.py —— 以任意一組資產產生「上列幾何／下列貼圖」的對照圖

`make_demo_figures.py --fig 16` 的資產清單寫死於其 OUTFITS，輸出檔名亦固定。
本腳本沿用其渲染與驅動函式，僅把資產清單與輸出檔名改為命令列參數，
故可為不同的資產組合各自出圖，不需來回修改 OUTFITS。

**渲染與驅動邏輯一律 import 自 make_demo_figures，不另寫一份。**
此前已有數次因兩份副本各自演進而產生的不一致（drive 的簽章、貼圖路徑），
故此處只做參數化，不複製實作。

為何需要兩列
============
下列為貼圖後的彩現。語意區域依**人體部位**而非衣物邊界劃分，故裙襬以下
裸露的小腿會被塗上服裝顏色——單看下列無法區分「及膝裙」與「長褲」。
上列的法向量著色呈現未經貼圖的幾何，服裝類型須以其為準（附錄 C）。

用法
====
  conda activate chupa
  cd ~/projects/chupa

  # 兩件補充服裝（黑色西裝、白色羽絨大衣）
  python make_outfit_grid.py \
      --assets demo_work/E1_black_suit:black suit \
               demo_work/E2_white_down_coat:white down coat \
      --out figures/fig_extra_outfits.png

  # 五件一起（三件原有＋兩件補充）；格數多時建議加 --split
  python make_outfit_grid.py --preset all --split 3 \
      --out figures/fig_outfits_all.png

環境：chupa（需 torch、nvdiffrast、opencv）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CANVAS = 512

# 便利預設。冒號後為圖上標籤。
PRESETS = {
    "extra": [
        ("demo_work/E1_black_suit", "black suit"),
        ("demo_work/E2_white_down_coat", "white down coat"),
    ],
    "all": [
        ("demo_work/A1_red_dress_v2", "red dress"),
        ("demo_work/A2_navy_coat_q50", "navy coat + khaki pants"),
        ("demo_work/A3_green_hood_q50", "green hoodie + jeans"),
        ("demo_work/E1_black_suit", "black suit"),
        ("demo_work/E2_white_down_coat", "white down coat"),
    ],
}


def parse_assets(items):
    out = []
    for it in items:
        if ":" in it:
            w, lb = it.split(":", 1)
        else:
            w, lb = it, Path(it).name
        out.append((w.strip(), lb.strip()))
    return out


def main():
    ap = argparse.ArgumentParser(description="產生服裝對照圖（幾何／貼圖兩列）")
    ap.add_argument("--assets", nargs="*", default=[],
                    help="資產目錄[:標籤]，可多個")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="")
    ap.add_argument("--out", default="figures/fig_outfits.png")
    ap.add_argument("--bind", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    ap.add_argument("--model-dir", default="src/smpl_related/models")
    ap.add_argument("--temporal", default="temporal_eval")
    ap.add_argument("--pose-from", default="",
                    help="借用哪一段動作的姿態；省略則用綁定 A-pose")
    ap.add_argument("--split", type=int, default=0,
                    help="每塊幾格；預設全部排成一列")
    ap.add_argument("--size", type=int, default=CANVAS)
    ap.add_argument("--back-view", action="store_true")
    ap.add_argument("--no-flip-y", action="store_true")
    args = ap.parse_args()

    if args.preset:
        items = PRESETS[args.preset]
    elif args.assets:
        items = parse_assets(args.assets)
    else:
        sys.exit("[錯誤] 請以 --assets 或 --preset 指定資產")

    import cv2
    import nvdiffrast.torch as dr
    from make_demo_figures import (load_asset, drive, make_view, render,
                                   label_strip)
    from drive_avatar import (load_bind, build_smplx, global_transforms,
                              motion_to_full_pose)

    bind = load_bind(args.bind)
    J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(
        global_transforms(J_rest, bind["full_pose"], parents))

    # 各套服裝**共用同一姿態**。若各自擺不同姿勢，讀者無從分辨差異
    # 來自服裝本身或姿態。
    if args.pose_from:
        poses = np.load(Path(args.temporal) / f"pose_{args.pose_from}.npz")["poses"]
        full = motion_to_full_pose(poses, bind["full_pose"])
        pose = full[len(full) // 2]
        print(f"[INFO] 共用 {args.pose_from} 之第 {len(full) // 2} 影格姿態")
    else:
        pose = bind["full_pose"]
        print("[INFO] 共用綁定姿勢（A-pose）")

    assets, driven, labels, missing = [], [], [], []
    for work, label in items:
        p = ROOT / work
        if not (p / "geometry.npz").exists():
            missing.append(work)
            continue
        a = load_asset(work)
        assets.append(a)
        driven.append(drive(a, bind, J_rest, parents, pose, G_bind_inv))
        labels.append(label)
    if missing:
        print(f"[警告] 以下資產缺少 geometry.npz，已略過：{'、'.join(missing)}")
    if not driven:
        sys.exit("[錯誤] 沒有可用的資產")

    # 各格共用取景，各套服裝的體型比例才可直接比對
    view = make_view(driven)
    ctx = dr.RasterizeCudaContext()

    def shot(v, a, mode):
        return cv2.cvtColor(
            render(v, a, view, ctx, size=args.size, mode=mode,
                   flip_y=not args.no_flip_y, back_view=args.back_view),
            cv2.COLOR_RGB2BGR)

    n = args.split if args.split and args.split > 0 else len(driven)
    idx = list(range(len(driven)))
    chunks = [idx[i:i + n] for i in range(0, len(idx), n)]

    blocks = []
    for ch in chunks:
        geo = [shot(driven[i], assets[i], "normal") for i in ch]
        tex = [shot(driven[i], assets[i], "texture") for i in ch]
        lab = [labels[i] for i in ch]
        blocks.append(np.vstack([
            label_strip(geo, ["geometry (normals)"] + [""] * (len(ch) - 1)),
            label_strip(tex, lab),
        ]))

    w = max(b.shape[1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[1] < w:
            b = np.hstack([b, np.full((b.shape[0], w - b.shape[1], 3),
                                      255, np.uint8)])
        padded.append(b)
        padded.append(np.full((10, w, 3), 255, np.uint8))
    fig = np.vstack(padded[:-1])

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(op), fig)
    print(f"[完成] {op}（{len(driven)} 套服裝、{len(chunks)} 塊、"
          f"每塊至多 {n} 格）")

    if n >= 5:
        print("\n  [提醒] 五格並排時每格於論文頁寬下僅約 3 公分。"
              "\n         上列用以判定服裝類型（西裝的翻領、羽絨大衣的蓬鬆"
              "輪廓），該尺寸下不易辨識。建議加 --split 3。")
    print("\n圖說須載明：各格共用同一體型、同一姿態與同一取景；"
          "\n上列為未貼圖之幾何，服裝類型應以其為準。")


if __name__ == "__main__":
    main()
