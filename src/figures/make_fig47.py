r"""
make_fig47.py —— 產生論文圖 4-7（ROMP 姿態估測與形狀估測的落差）

圖 4-7 為左右並排：左為輸入影像，右為 ROMP 輸出之 SMPL 網格疊合於原影像。
其論點為「姿態估測準確、形狀估測退化至平均體型」。

此圖成立的兩個前提，皆不可由圖面本身確認，故本腳本逐項處理：

  前提一：**受測者的體型確實明顯異於平均**。
          若挑到體型接近平均的人，ROMP 輸出平均體型會看起來剛好正確，
          該圖即證明不了任何事。此點腳本無法判定——須由使用者挑選片段，
          故預設先進入 `--scan` 模式列出各段供挑選，不直接出圖。

  前提二：**形狀參數確實趨近於零**（論文所載「各分量絕對值均小於 0.02」）。
          此點可量測。`--scan` 會逐段回報形狀參數的絕對值上限，使論文中的
          數字有據可查，而非僅憑單一影格的印象。

用法
====
  conda activate romp

  # 步驟一：掃描各段影片，回報形狀參數統計，並挑選要用的片段
  python make_fig47.py --scan --video-dir ~/projects/chupa/motion_testVideo

  # 步驟二：對選定的影片輸出候選影格一覽，挑一格姿態清楚者
  python make_fig47.py --sheet --video 3.mp4 --video-dir ~/...

  # 步驟三：出圖
  python make_fig47.py --video 3.mp4 --frame 42 --video-dir ~/... --out figures

環境：romp（需 simple-romp、opencv、numpy）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# 論文 4.4.1 節現載之實測上限（全部受測者、2,317 影格）。
# 初版曾寫 0.02，經全掃後修正為此值；掃描結果若超過，該句須再次更新。
BETA_CLAIM = 0.0443

SHEET_N = 12          # 候選影格一覽的格數
PANEL_W = 640         # 出圖時每一欄的寬度


def init_romp(render=True):
    try:
        import romp
    except ImportError:
        sys.exit("[錯誤] 找不到 romp。請先 conda activate romp")
    s = romp.main.default_settings
    s.calc_smpl = True
    s.render_mesh = render
    s.show = False
    # 單人影片。show_largest 可避免畫面中偶然出現他人時輸出順序不穩定
    if hasattr(s, "show_largest"):
        s.show_largest = True
    return romp.ROMP(s)


def iter_frames(video_path, stride=1, limit=None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[錯誤] 打不開影片：{video_path}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield i, frame
            if limit and (i // stride) + 1 >= limit:
                break
        i += 1
    cap.release()


def get_betas(out):
    """自 ROMP 輸出取形狀參數。不同版本的鍵名不一，故逐一嘗試。"""
    for k in ("smpl_betas", "betas", "smpl_shape"):
        if out is not None and k in out and len(out[k]):
            return np.asarray(out[k][0]).reshape(-1)
    return None


def scan(args):
    import cv2
    vdir = Path(os.path.expanduser(args.video_dir))
    vids = sorted(p for p in vdir.iterdir()
                  if p.suffix.lower() in (".mp4", ".mov", ".avi"))
    if not vids:
        sys.exit(f"[錯誤] {vdir} 下沒有影片")

    model = init_romp(render=False)
    print(f"\n逐段掃描形狀參數（每 {args.stride} 影格取一）\n")
    print(f"{'影片':<14}{'取樣':>6}{'偵測':>6}{'|β|最大':>10}"
          f"{'|β|平均':>10}  判定")
    worst = 0.0
    for vp in vids:
        mags, n = [], 0
        for _, frame in iter_frames(vp, args.stride, args.limit):
            n += 1
            b = get_betas(model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if b is not None:
                mags.append(np.abs(b).max())
        if not mags:
            print(f"{vp.name:<14}{n:>6}{0:>6}{'—':>10}{'—':>10}  無偵測結果")
            continue
        mx, mn = float(np.max(mags)), float(np.mean(mags))
        worst = max(worst, mx)
        verdict = "趨近於零" if mx < BETA_CLAIM else "**超過論文所載門檻**"
        print(f"{vp.name:<14}{n:>6}{len(mags):>6}{mx:>10.4f}{mn:>10.4f}"
              f"  {verdict}")

    print(f"\n全段最大 |β| = {worst:.4f}")
    if worst < BETA_CLAIM:
        print(f"  ✓ 論文 4.4.1 節「各分量絕對值均小於 {BETA_CLAIM}」與實測相符。")
    else:
        print(f"  ✗ 實測最大值 {worst:.4f} 已超過論文所載的 {BETA_CLAIM}。"
              f"\n     論文該句須改為實測值，或說明取樣範圍。")

    print("\n接下來請**自行挑選體型明顯異於平均的受測者所在片段**——"
          "\n此點無法由形狀參數判定（參數趨近於零正是本圖要呈現的問題），"
          "\n只能由妳確認影片中的人實際體型。選定後執行："
          "\n  python make_fig47.py --sheet --video <檔名> --video-dir ...")


def sheet(args):
    import cv2
    vp = Path(os.path.expanduser(args.video_dir)) / args.video
    if not vp.exists():
        vp = Path(os.path.expanduser(args.video))
    if not vp.exists():
        sys.exit(f"[錯誤] 找不到影片 {args.video}")

    frames = [(i, f) for i, f in iter_frames(vp, args.stride, SHEET_N)]
    if not frames:
        sys.exit("[錯誤] 讀不到影格")

    cells = []
    for i, f in frames:
        c = cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1])))
        cv2.putText(c, f"frame {i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cells.append(c)
    h = min(c.shape[0] for c in cells)
    cells = [c[:h] for c in cells]
    ncol = 4
    rows = [cells[i:i + ncol] for i in range(0, len(cells), ncol)]
    for r in rows:
        while len(r) < ncol:
            r.append(np.full_like(cells[0], 255))
    grid = np.vstack([np.hstack(r) for r in rows])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "fig_4-7_candidates.png"
    cv2.imwrite(str(p), grid)
    print(f"[完成] {p}")
    print("請挑一格**姿態清楚、全身入鏡、無遮擋**者，記下其 frame 編號，"
          "再執行：\n  python make_fig47.py --video "
          f"{args.video} --frame <編號> --video-dir ... --out figures")


def render(args):
    import cv2
    vp = Path(os.path.expanduser(args.video_dir)) / args.video
    if not vp.exists():
        vp = Path(os.path.expanduser(args.video))
    if not vp.exists():
        sys.exit(f"[錯誤] 找不到影片 {args.video}")

    # 循序讀到指定影格，不用 CAP_PROP_POS_FRAMES 跳轉——該跳轉在部分編碼下
    # 會落到鄰近的關鍵影格，使實際取得的影格與所指定者不同，而畫面上看不出來。
    cap = cv2.VideoCapture(str(vp))
    frame, k = None, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k == args.frame:
            frame = fr
            break
        k += 1
    cap.release()
    if frame is None:
        sys.exit(f"[錯誤] 讀不到第 {args.frame} 影格（全片共 {k} 影格）")

    model = init_romp(render=True)
    out = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if out is None or "smpl_thetas" not in out or len(out["smpl_thetas"]) == 0:
        sys.exit(f"[錯誤] 第 {args.frame} 影格未偵測到人，請改選其他影格")

    b = get_betas(out)
    if b is not None:
        print(f"[INFO] 本影格之形狀參數 |β| 最大 {np.abs(b).max():.4f}")
        print(f"       各分量：{np.array2string(b, precision=4, max_line_width=100)}")
        if np.abs(b).max() > BETA_CLAIM:
            print(f"  [警告] 本影格的 |β| 為 {np.abs(b).max():.4f}，"
                  f"高於論文 4.4.1 節所載的全段上限 {BETA_CLAIM}。"
                  f"\n         該上限須更新，否則圖說與正文不符。")
    else:
        print("  [警告] 無法自輸出取得形狀參數，圖說中的數值須另行查證。")

    if "rendered_image" not in out:
        sys.exit("[錯誤] ROMP 未回傳 rendered_image。"
                 "請確認 simple-romp 版本支援 settings.render_mesh，"
                 "或改用其 CLI 的 --render_mesh 產生疊合圖。")
    overlay = out["rendered_image"]
    if overlay.ndim == 3 and overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # ROMP 某些版本的 rendered_image 為「原影像 ∥ 網格」的並排圖，而非疊合。
    # 若直接整張放入，左半會與本圖的 (a) 格重複，且圖說所稱之「疊合」與事實
    # 不符。此處依寬高比判定並只取網格那半。
    ar_in = frame.shape[1] / frame.shape[0]
    ar_ov = overlay.shape[1] / overlay.shape[0]
    side_by_side = ar_ov > ar_in * 1.6
    if side_by_side:
        print("[INFO] ROMP 回傳的是並排圖（原影像 ∥ 網格），已取右半的網格。"
              "\n       圖說不可寫「疊合」——本圖為並排比較。")
        overlay = overlay[:, overlay.shape[1] // 2:]
    else:
        print("[INFO] ROMP 回傳的是疊合圖（網格繪於原影像之上）。")

    def fit(img):
        s = PANEL_W / img.shape[1]
        return cv2.resize(img, (PANEL_W, int(img.shape[0] * s)))

    left, right = fit(frame), fit(overlay)
    h = min(left.shape[0], right.shape[0])
    left, right = left[:h], right[:h]
    gap = np.full((h, 12, 3), 255, np.uint8)
    fig = np.hstack([left, gap, right])

    band = 40
    b_img = np.full((band, fig.shape[1], 3), 255, np.uint8)
    lbl_b = "(b) ROMP SMPL mesh" if side_by_side else "(b) ROMP SMPL overlay"
    cv2.putText(b_img, "(a) input frame", (10, band - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(b_img, lbl_b, (PANEL_W + 22, band - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    fig = np.vstack([fig, b_img])

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)

    # (a) 與 (b) 必須出自同一影格。此處另存一份實際送入 ROMP 的影像供核對——
    # 若其與圖中的 (a) 格不同，即表示取影格的環節有誤。
    dbg = od / "fig_4-7_frame_fed_to_romp.png"
    cv2.imwrite(str(dbg), frame)
    print(f"[INFO] 實際送入 ROMP 的影像另存於 {dbg}，"
          "請確認其與圖中的 (a) 格一致。")
    p = od / "fig_4-7_romp_shape.png"
    cv2.imwrite(str(p), fig)
    print(f"\n[完成] {p}（{vp.name} 第 {args.frame} 影格）")
    print("\n出圖後請自行確認：影像中人物的體型是否明顯異於平均？"
          "\n若否，本圖無法支持 4.4.1 節的論點，須改用其他片段。")


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-7")
    ap.add_argument("--video-dir", default="~/projects/chupa/motion_testVideo")
    ap.add_argument("--video", default="")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=20,
                    help="--scan 時每段最多取樣的影格數")
    args = ap.parse_args()

    if args.scan:
        scan(args)
    elif args.sheet:
        if not args.video:
            sys.exit("[錯誤] --sheet 須以 --video 指定影片")
        sheet(args)
    elif args.video and args.frame >= 0:
        render(args)
    else:
        print("請先執行 --scan 挑選片段：\n"
              "  python make_fig47.py --scan --video-dir "
              f"{args.video_dir}")


if __name__ == "__main__":
    main()
