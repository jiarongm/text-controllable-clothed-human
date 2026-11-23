r"""
prep_romp_motion.py —— 為新影片產出 ROMP 輸出，並順便量測形狀參數

一支腳本同時滿足三處需求，避免同一段影片被跑三次 ROMP：

  1. **形狀參數統計**（4.4.1 節）——逐格的 |β|，與既有受測者比較。
  2. **render_overlay 所需的 motion 目錄**——每影格一個 npz，含 `smpl_thetas`、
     `joints`、`pj2d_org`。`run_romp_batch.py` 只存 thetas，缺二維關節，
     無法用於合成。
  3. **drive_avatar 所需的姿態序列**——另存一份 `pose_<名稱>.npz`。

為何不沿用 run_romp_batch.py
============================
該腳本為 4.3.5 節的連續動作評估而寫，每段上限 200 影格且只保留 thetas。
本腳本不設上限、保留合成所需的全部欄位，且**不寫入 temporal_eval/**——
新素材若混入該目錄，4.3.5 節的影格計數與六段動作的定義都會被更動。

用法
====
  conda activate romp
  python prep_romp_motion.py --video ~/projects/chupa/motion_largebody/<檔名>.mp4 \
      --name largebody --out motion_out

  # 只看形狀參數，不寫檔
  python prep_romp_motion.py --video ... --name largebody --scan-only

輸出
====
  motion_out/largebody/frame_000000.npz ...   ← render_overlay --motion 指向此目錄
  motion_out/pose_largebody.npz               ← drive_avatar / make_fig413 可用
  motion_out/largebody_betas.csv              ← 逐格形狀參數，供後續統計

環境：romp
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# render_overlay._extract 以這些鍵名尋找欄位，故存檔時務必沿用，
# 不可自行改名——改名後 render_overlay 會靜默地取不到二維關節，
# 退回只用三維對位，衣服會整體偏移而不報錯。
KEY_POSE = "smpl_thetas"
KEY_J3D = "joints"
KEY_J2D = "pj2d_org"
KEY_BETA = "smpl_betas"

# 與既有六段受測者的比較基準（見 4.4.1 節之全掃結果）
REF_LABEL = "既有六段（接近平均體型）"
REF_MAX, REF_MED, REF_N = 0.0443, 0.0128, 1613


def init_romp():
    try:
        import romp
    except ImportError:
        sys.exit("[錯誤] 找不到 romp。請先 conda activate romp")
    s = romp.main.default_settings
    s.calc_smpl = True
    s.render_mesh = False
    s.show = False
    if hasattr(s, "show_largest"):
        s.show_largest = True
    return romp.ROMP(s)


def take(out, keys, person=0):
    for k in keys:
        if k in out and out[k] is not None and len(out[k]):
            v = np.asarray(out[k])
            return v[min(person, len(v) - 1)]
    return None


def main():
    ap = argparse.ArgumentParser(description="為新影片產出 ROMP 輸出與形狀統計")
    ap.add_argument("--video", required=True)
    ap.add_argument("--name", required=True, help="此段動作的名稱，例如 largebody")
    ap.add_argument("--out", default="motion_out")
    ap.add_argument("--person", type=int, default=0)
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()

    import cv2

    vp = Path(os.path.expanduser(args.video))
    if not vp.exists():
        sys.exit(f"[錯誤] 找不到影片 {vp}")

    od = Path(args.out)
    fd = od / args.name
    if not args.scan_only:
        fd.mkdir(parents=True, exist_ok=True)

    model = init_romp()
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        sys.exit(f"[錯誤] 打不開 {vp}")

    poses, betas, rows = [], [], []
    n_total = n_miss = n_multi = n_no2d = 0
    k_out = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        out = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if out is None or "smpl_thetas" not in out or len(out["smpl_thetas"]) == 0:
            n_miss += 1
            continue
        if len(out["smpl_thetas"]) > 1:
            n_multi += 1

        th = take(out, ["smpl_thetas"], args.person)
        j3 = take(out, ["joints", "j3d", "smpl_joints"], args.person)
        j2 = take(out, ["pj2d_org", "pj2d", "joints_2d"], args.person)
        bt = take(out, ["smpl_betas", "betas"], args.person)
        if j2 is None:
            n_no2d += 1

        poses.append(np.asarray(th).reshape(-1))
        if bt is not None:
            b = np.asarray(bt).reshape(-1)
            betas.append(b)
            rows.append([n_total - 1, float(np.abs(b).max())] +
                        [float(x) for x in b])

        if not args.scan_only:
            d = {KEY_POSE: np.asarray(th)}
            if j3 is not None:
                d[KEY_J3D] = np.asarray(j3)
            if j2 is not None:
                d[KEY_J2D] = np.asarray(j2)
            if bt is not None:
                d[KEY_BETA] = np.asarray(bt)
            np.savez_compressed(fd / f"frame_{k_out:06d}.npz", **d)
            k_out += 1
    cap.release()

    if not poses:
        sys.exit("[錯誤] 全片未偵測到人，此段不可用")

    miss_rate = n_miss / max(n_total, 1)
    print(f"\n影片：{vp.name}")
    print(f"  讀入 {n_total} 影格，可用 {len(poses)}，"
          f"偵測失敗 {n_miss}（{100 * miss_rate:.1f}%）")
    if n_multi:
        print(f"  [警告] {n_multi} 個影格偵測到多人，已取第 {args.person} 個。"
              "若對象不固定，對位會在人之間跳動——請確認畫面中僅有一人，"
              "或以 --person 指定。")
    if n_no2d:
        print(f"  [警告] {n_no2d} 個影格缺二維關節（{KEY_J2D}），"
              "\n         render_overlay 於該處只能以三維對位，衣服可能整體偏移。")
    if miss_rate > 0.05:
        print(f"  [警告] 偵測失敗率偏高。失敗影格被跳過會使動作在該處接起，"
              "產生不存在的跳動。")

    # ---- 形狀參數 ----
    if betas:
        B = np.stack(betas)
        per_frame_max = np.abs(B).max(axis=1)
        mx, med = float(per_frame_max.max()), float(np.median(per_frame_max))
        print(f"\n形狀參數（n = {len(B)} 影格）")
        print(f"  |β| 最大 {mx:.4f}　中位數 {med:.4f}")
        print(f"  各分量的絕對值上限：")
        print("    " + np.array2string(np.abs(B).max(axis=0), precision=4,
                                       max_line_width=100))
        print(f"\n對照 {REF_LABEL}：最大 {REF_MAX:.4f}、中位數 {REF_MED:.4f}"
              f"、n = {REF_N}")
        ratio = mx / REF_MAX if REF_MAX else float("nan")
        print(f"  本段最大值為其 {ratio:.2f} 倍")
        if mx < REF_MAX * 2:
            print("\n  → 體型明顯不同的受測者，其形狀參數與接近平均體型者"
                  "**處於同一量級**。"
                  "\n    此直接支持「估測器的輸出不隨實際體型改變」，"
                  "較原本僅能主張的「值域過窄」更強。")
        else:
            print("\n  → 本段的形狀參數明顯大於既有受測者。"
                  "\n    這表示估測器**並非完全退化**，只是靈敏度不足；"
                  "\n    4.4.1 節「不具區分能力」之措辭須據此修正。")
        print("\n  （上述為提示，實際結論請以完整數據判斷）")

        if not args.scan_only:
            import csv
            cp = od / f"{args.name}_betas.csv"
            with open(cp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame", "abs_max"] +
                           [f"b{i}" for i in range(B.shape[1])])
                w.writerows(rows)
            print(f"\n[完成] 逐格形狀參數：{cp}")
    else:
        print("\n[警告] 未能自輸出取得形狀參數")

    if not args.scan_only:
        pp = od / f"pose_{args.name}.npz"
        np.savez_compressed(pp, poses=np.stack(poses))
        print(f"[完成] 姿態序列：{pp}（{len(poses)} 影格）")
        print(f"[完成] 逐格 ROMP 輸出：{fd}/（{k_out} 個檔）")
        print(f"\n接續："
              f"\n  圖 4-7   python make_fig47.py --sheet --video {vp.name} "
              f"--video-dir {vp.parent}"
              f"\n  合成     python render_overlay.py --video {vp} "
              f"--motion {fd} --work <資產目錄> \\"
              f"\n             --bind-pkl <對應的 bind.pkl> --out /tmp/overlay_{args.name}.mp4")


if __name__ == "__main__":
    main()
