r"""
make_driving_video.py —— 將即時姿態驅動展示錄成影片

論文圖 4-16（舊編號 4-15）為五個影格的靜態圖；本腳本輸出同一段動作的
完整影片，供口試簡報使用。

版面與靜態圖一致：上排為輸入影片之影格，下排為同一影格所驅動之著衣網格。

兩項必須與靜態圖一致的設定
==========================
1. **下排各格共用同一取景。** 若逐影格依當下的包圍盒取景，人體會隨動作
   忽大忽小，看起來像相機在推拉而非人在動——靜態圖的圖說已就此說明，
   影片若不比照，兩者會呈現不同的視覺行為。本腳本先掃過全段求取景，
   再以該值固定視野。
2. **深度符號。** SMPL-X 人體朝 +Z，而 nvdiffrast 保留 NDC z 較小者，
   故須取負，否則渲出的是背面（此錯誤曾實際發生於靜態圖）。

本腳本沿用 `make_demo_figures.py` 的 `make_view`、`render` 與驅動邏輯，
不另寫一份——對位與渲染只應存在於一處。

用法
====
  conda activate chupa
  cd ~/projects/chupa

  python make_driving_video.py --out figures/fig_4-16_driving.mp4

  # 換動作段與對應影片（1=shoulder、2=elbow、3=run、4=legraise、5=twist）
  python make_driving_video.py --motion run --video-index 3 \
      --out figures/fig_4-16_driving_run.mp4

  # 只輸出下排（無輸入影片對照），適合放簡報
  python make_driving_video.py --mesh-only --out figures/avatar_only.mp4

  # 運行中換裝：第 60 格換西裝、第 130 格換羽絨大衣
  python make_driving_video.py \
      --work demo_work/A2_navy_coat_q50 \
      --swap-to demo_work/E1_black_suit demo_work/E2_white_down_coat \
      --swap-at 60 130 --swap-label \
      --out figures/outfit_swap.mp4

環境：chupa（需 torch、nvdiffrast、opencv）
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CANVAS = 512


def main():
    ap = argparse.ArgumentParser(description="錄製即時驅動展示影片")
    ap.add_argument("--work", default="demo_work/A2_navy_coat_q50")
    ap.add_argument("--bind", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    ap.add_argument("--model-dir", default="src/smpl_related/models")
    ap.add_argument("--temporal", default="temporal_eval")
    ap.add_argument("--motion", default="shoulder")
    ap.add_argument("--video-index", default="1")
    ap.add_argument("--video-dir", default="~/projects/chupa/motion_testVideo")
    ap.add_argument("--out", default="figures/fig_4-16_driving.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=CANVAS)
    ap.add_argument("--mesh-only", action="store_true",
                    help="只輸出驅動結果，不併入輸入影片")
    ap.add_argument("--layout", choices=["h", "v"], default="h",
                    help="h＝左右並排（預設，適合投影）、v＝上下堆疊"
                         "（與靜態圖 4-16 的版面一致）")
    ap.add_argument("--swap-to", nargs="*", default=[],
                    help="換裝用的資產目錄，可多個，依序替換")
    ap.add_argument("--swap-at", nargs="*", type=int, default=[],
                    help="於第幾影格替換，數量須與 --swap-to 相同")
    ap.add_argument("--swap-label", action="store_true",
                    help="於替換後數格於畫面上標示（便於觀眾注意到）")
    ap.add_argument("--limit", type=int, default=0, help="只處理前 N 影格")
    ap.add_argument("--no-flip-y", action="store_true")
    args = ap.parse_args()

    import cv2
    import nvdiffrast.torch as dr
    from make_demo_figures import load_asset, drive, make_view, render
    from drive_avatar import (load_bind, build_smplx, global_transforms,
                              motion_to_full_pose)

    pose_p = Path(args.temporal) / f"pose_{args.motion}.npz"
    if not pose_p.exists():
        sys.exit(f"[錯誤] 找不到 {pose_p}，請先執行 run_romp_batch.py")
    poses = np.load(pose_p)["poses"]

    vp = Path(os.path.expanduser(args.video_dir)) / f"{args.video_index}.mp4"
    if not args.mesh_only and not vp.exists():
        sys.exit(f"[錯誤] 找不到輸入影片 {vp}（或改用 --mesh-only）")

    work = Path(args.work)
    if not (work / "geometry.npz").exists():
        sys.exit(f"[錯誤] {work} 缺少 geometry.npz")

    # ---- 換裝設定 ----
    #
    # 3.6.3 節之原子性替換：換裝時**幾何、UV、蒙皮權重與貼圖必須整組一起換**。
    # 若只換貼圖而沿用舊的幾何與權重，畫面不會報錯，但貼圖會對應到錯誤的
    # 表面——此為該節所述之關鍵限制，故此處以整個 asset 字典為替換單位。
    if len(args.swap_to) != len(args.swap_at):
        sys.exit(f"[錯誤] --swap-to 有 {len(args.swap_to)} 個、"
                 f"--swap-at 有 {len(args.swap_at)} 個，數量須相同")
    swaps = sorted(zip(args.swap_at, args.swap_to))
    for at, to in swaps:
        p = ROOT / to
        if not (p / "geometry.npz").exists():
            sys.exit(f"[錯誤] 換裝目標 {to} 缺少 geometry.npz")
    if swaps:
        print("[INFO] 換裝排程：" + "、".join(
            f"第 {at} 格 → {Path(to).name}" for at, to in swaps))

    asset = load_asset(str(work))
    bind = load_bind(args.bind)
    J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
    # 綁定姿勢的逆變換只需計算一次，由呼叫端傳入 drive()。
    # make_demo_figures.drive 的簽章即為如此，不可省略此參數。
    G_bind_inv = np.linalg.inv(
        global_transforms(J_rest, bind["full_pose"], parents))
    full = motion_to_full_pose(poses, bind["full_pose"])
    n = len(full) if not args.limit else min(args.limit, len(full))
    print(f"[INFO] 動作 {args.motion}：{len(full)} 影格，輸出 {n} 影格")

    # ---- 共用取景：先掃過全段，取其聯集 ----
    # 逐影格取景會使人體隨動作忽大忽小，讀者會誤讀為深度方向的位移。
    # 此與靜態圖（圖 4-16）的處理一致，兩者的視覺行為方能相符。
    print("[INFO] 掃描全段以決定共用取景 ...")
    step = max(1, n // 60)                 # 取樣約 60 格即足以涵蓋極值
    # 換裝時各套服裝的體積不同（如羽絨大衣明顯厚於 T 恤），故取景須涵蓋
    # **全部**將出現的資產，否則換裝後可能超出畫面，或畫面突然改變比例，
    # 使觀眾誤以為是尺度跳動而非換裝。
    scan_assets = [asset] + [load_asset(to) for _, to in swaps]
    sampled = []
    for a in scan_assets:
        sampled += [drive(a, bind, J_rest, parents, full[i], G_bind_inv)
                    for i in range(0, n, step)]
    view = make_view(sampled)
    print(f"[INFO] 取景已固定（{len(scan_assets)} 套服裝 × "
          f"約 {len(sampled) // len(scan_assets)} 影格）")

    ctx = dr.RasterizeCudaContext()

    cap = None
    if not args.mesh_only:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            sys.exit(f"[錯誤] 打不開 {vp}")

    # 版面：靜態圖採上下兩排（頁面為直式），影片預設左右並排（投影為橫式）。
    # 二者不同為刻意選擇，非疏漏；圖說若提及版面，須分別敘述。
    S = args.size
    if args.mesh_only:
        out_w, out_h = S, S
    elif args.layout == "h":
        out_w, out_h = S * 2, S
    else:
        out_w, out_h = S, S * 2
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(op), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (out_w, out_h))
    if not vw.isOpened():
        sys.exit(f"[錯誤] 無法建立影片 {op}")

    n_missing = 0
    pending = list(swaps)
    swap_done = []                 # (影格, 名稱, 耗時毫秒)
    for i in range(n):
        # 原子性替換：整組換掉，換完才進入下一格的驅動
        while pending and i >= pending[0][0]:
            at, to = pending.pop(0)
            t0 = time.time()
            asset = load_asset(to)
            dt = (time.time() - t0) * 1000.0
            swap_done.append((i, Path(to).name, dt))
            print(f"  [第 {i} 格] 換裝 → {Path(to).name}"
                  f"（載入 {dt:.1f} ms）")

        v = drive(asset, bind, J_rest, parents, full[i], G_bind_inv)
        low = cv2.cvtColor(
            render(v, asset, view, ctx, size=S,
                   flip_y=not args.no_flip_y), cv2.COLOR_RGB2BGR)

        if args.swap_label and swap_done:
            last_i, last_name, _ = swap_done[-1]
            if 0 <= i - last_i < 20:        # 換裝後 20 格內標示
                cv2.putText(low, f"swapped: {last_name}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                            cv2.LINE_AA)

        if args.mesh_only:
            vw.write(low)
        else:
            ok, fr = cap.read()
            if not ok:
                n_missing += 1
                fr = np.zeros((S, S, 3), np.uint8)
            else:
                h, w = fr.shape[:2]
                s = min(h, w)
                fr = fr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
                fr = cv2.resize(fr, (S, S), interpolation=cv2.INTER_AREA)
            # 左／上為輸入影片，右／下為驅動結果——順序與靜態圖一致
            vw.write(np.hstack([fr, low]) if args.layout == "h"
                     else np.vstack([fr, low]))

        if (i + 1) % 30 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}")
    if cap is not None:
        cap.release()
    vw.release()

    print(f"\n[完成] {op}（{n} 影格 @ {args.fps} fps，{n / args.fps:.1f} 秒）")

    if swap_done:
        print("\n換裝記錄：")
        for at, name, dt in swap_done:
            print(f"  第 {at} 格（約 {at / args.fps:.1f} 秒）→ {name}"
                  f"，資產載入 {dt:.1f} ms")
        print("\n  **此處的載入耗時不可直接引為 4.7.3 節的 31.29 毫秒。**"
              "\n  該節之量測為 30 次重複、熱快取、拆解為貼圖解碼／幾何載入／"
              "\n  蒙皮權重／GPU 上傳四段；此處為單次、未控制快取狀態，"
              "\n  且不含 GPU 上傳（離線渲染每格自行上傳）。"
              "\n  本影片所展示者為**替換的原子性與視覺連續性**，非其延遲。")
    if n_missing:
        print(f"  [警告] 輸入影片有 {n_missing} 個影格讀不到，已以黑畫面填補。"
              "\n         姿態序列與影片的影格數可能不一致——"
              "姿態取自 ROMP，偵測失敗的影格會被跳過，"
              "\n         故兩者的第 i 格未必對應同一時刻。"
              "**上下兩排的對應關係須據此保留**。")
    print("\n注意：本影片為**離線渲染**，非即時執行的錄影。"
          "\n其每影格的渲染時間與 4.7 節所量測的驅動耗時無關——"
          "後者不含光柵化與合成（表 4-23 之註記）。"
          "\n簡報時若以本影片說明即時性，須一併說明此點。")


if __name__ == "__main__":
    main()
