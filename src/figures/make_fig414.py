r"""
make_fig414.py —— 由 render_overlay.py 的三格對照影片組出論文圖 4-14

圖 4-14 為 3 列 × 3 格：每列一個姿勢，三格依序為
  (a) 輸入之原始影片影格
  (b) 以語意遮罩將服裝疊回原影像
  (c) 對應之完整著衣人體模型

`render_overlay.py` 已輸出這三格並排的影片，故本腳本不重做對位，只負責
挑影格與排版——**對位邏輯只應存在於一處**，複製一份到此處遲早會與原件
不同步。

為何不自動挑三格
================
論文正文明確指涉「雙手上舉的姿勢（圖 4-14 第一列）」，且 4.6.2 節的三項
失敗情形各自繫於特定姿勢。影格的選擇因此承載論證，不可由程式任意取樣。
本腳本提供 `--sheet` 列出候選供挑選，再以 `--rows` 指定。

腳本會檢查三列是否確實為不同姿勢——若兩列過於相似，該列便不提供新資訊，
而讀者無從察覺（圖說仍宣稱「三個不同的姿勢」）。

用法
====
  # 甲、已有先前挑好的三張三格影像（thesis_figures/fig4-15_frame{1,2,3}.png）
  python make_fig414.py \
      --frames thesis_figures/fig4-15_frame1.png,thesis_figures/fig4-15_frame2.png,thesis_figures/fig4-15_frame3.png \
      --out figures

  # 乙、自三格對照影片重新挑選
  python make_fig414.py --video thesis_figures/fig4-15_overlay.mp4 --sheet
  python make_fig414.py --video thesis_figures/fig4-15_overlay.mp4 \
      --rows 48,120,205 --out figures

註：`thesis_figures/` 下的檔名為舊版圖號（當時的圖 4-15 即現行的圖 4-14），
    內容無誤，僅命名未隨第四章重排而更新。

環境：chupa（需 opencv、numpy）
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

SHEET_N = 16          # 候選一覽的格數
ROW_W = 1200          # 出圖時每列的寬度

# 姿勢差異的度量：**剪影的 1 − IoU**。
#
# 初版採「整格像素的平均絕對差」，並憑空設門檻 0.02——此為錯誤的做法，
# 有兩個問題：其一，人體僅佔畫面一小部分，背景會稀釋差異，故數值大小
# 不具可解讀性；其二，門檻無任何校準依據。
#
# 改以剪影的交集聯集比，數值即「兩個姿勢的輪廓有多少不重疊」，與人體在
# 畫面中所佔比例無關。門檻則**不再憑空指定**：以 --calibrate 掃過影片中
# 的實際影格對，取其分布後再判斷。下列預設值僅為占位，校準後應覆寫。
POSE_DIFF_MIN = 0.15
CALIBRATED = False          # 經 --calibrate 確認後方可改為 True


def read_frames(path, idxs):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"[錯誤] 打不開 {path}")
    want, got, k = set(int(i) for i in idxs), {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            got[k] = fr
        k += 1
        if len(got) == len(want):
            break
    cap.release()
    return got, k


def n_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def third_panel(fr):
    """取右側三分之一（純模型那格）。該格無背景，最適合比較姿勢差異。"""
    w = fr.shape[1]
    return fr[:, 2 * (w // 3):]


def silhouette(panel, tol=28):
    """自純模型格取出人體剪影。

    背景色由四條邊界的眾數推定，不預設為黑或白——render_overlay 的背景色
    可能隨版本而異，寫死會在無聲無息之下把整格都判為人體。
    """
    p = panel.astype(np.int16)
    border = np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]], axis=0)
    # 以 32 階量化後取眾數，避免壓縮雜訊使眾數落在單一像素值上
    q = (border // 32).astype(np.int32)
    key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    k = vals[np.argmax(cnt)]
    bg = np.array([(k // 1024) * 32 + 16, ((k // 32) % 32) * 32 + 16,
                   (k % 32) * 32 + 16], np.int16)
    return np.linalg.norm(p - bg.reshape(1, 1, 3), axis=2) > tol


def pose_diff(pa, pb):
    """1 − 剪影 IoU。0 表示輪廓完全重合，1 表示毫無交集。"""
    h = min(pa.shape[0], pb.shape[0])
    w = min(pa.shape[1], pb.shape[1])
    a = silhouette(pa[:h, :w])
    b = silhouette(pb[:h, :w])
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return 1.0 - float(np.logical_and(a, b).sum()) / float(union)


def calibrate(args):
    """掃過影片，取影格對的 1 − IoU 分布，作為門檻的經驗依據。

    同時回報**相鄰影格**的差異——該值對應「幾乎同一個姿勢」，
    即此度量在本影片上的雜訊底線。
    """
    import cv2
    total = n_frames(args.video)
    if total < 4:
        sys.exit("[錯誤] 影格太少，無從校準")
    k = min(args.calib_n, total)
    idxs = sorted(set(np.linspace(0, total - 1, k).astype(int).tolist()))
    # 另取若干組相鄰影格作為雜訊底線
    adj = [i for i in idxs[:-1] if i + 1 < total]
    got, _ = read_frames(args.video, idxs + [i + 1 for i in adj])

    panels = {i: third_panel(got[i]) for i in got}
    base = [pose_diff(panels[i], panels[i + 1])
            for i in adj if i in panels and (i + 1) in panels]
    pairs = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            ia, ib = idxs[a], idxs[b]
            if ia in panels and ib in panels:
                pairs.append(pose_diff(panels[ia], panels[ib]))

    base = [x for x in base if np.isfinite(x)]
    pairs = [x for x in pairs if np.isfinite(x)]
    if not pairs:
        sys.exit("[錯誤] 無法取得任何影格對")

    print(f"\n校準（影片共 {total} 影格，取樣 {len(idxs)} 格）")
    print(f"  相鄰影格（≈同一姿勢，雜訊底線）：中位數 "
          f"{float(np.median(base)):.4f}，最大 {max(base):.4f}  n={len(base)}")
    qs = np.percentile(pairs, [5, 25, 50, 75, 95])
    print(f"  任意兩格：P5 {qs[0]:.4f}  P25 {qs[1]:.4f}  中位數 {qs[2]:.4f}  "
          f"P75 {qs[3]:.4f}  P95 {qs[4]:.4f}  n={len(pairs)}")
    print(f"  全距：{min(pairs):.4f} – {max(pairs):.4f}")
    # 門檻應對應所欲排除的虛無假設。圖說宣稱的是「三個**不同的**姿勢」，
    # 而非「三個差異特別大的姿勢」，故須排除者為「實為同一姿勢」。
    # 該虛無假設的經驗分布即上方的雜訊底線，門檻應取其上界，
    # 而非隨機兩格的中位數——後者會要求「比一半的隨機配對更不同」，
    # 那是比圖說更強的宣稱，據以否決正確的選圖並不正當。
    floor_max = max(base) if base else float("nan")
    print(f"\n  建議門檻 {floor_max:.3f}（雜訊底線之上界）")
    print(f"    圖說所宣稱者為「三個不同的姿勢」，故須排除的是「實為同一"
          f"姿勢」。\n    相鄰影格即該情形的經驗分布，其上界為 "
          f"{floor_max:.4f}；超過此值，即可與「同一姿勢」區辨。")
    print(f"    較嚴格的選擇為隨機兩格的 P5（{qs[0]:.3f}），"
          f"其意為「至少與 95% 的隨機配對一樣不同」。")
    print(f"    **不宜**採隨機兩格的中位數（{qs[2]:.3f}）——"
          f"那要求「比一半的隨機配對更不同」，"
          f"\n    是比圖說更強的宣稱，據以否決正確的選圖並不正當。")
    print(f"\n  以 --min-diff {floor_max:.3f} 傳入；"
          f"確認後可將 POSE_DIFF_MIN 設為該值並把 CALIBRATED 改為 True。")


def sheet(args):
    import cv2
    total = n_frames(args.video)
    if total <= 0:
        sys.exit("[錯誤] 讀不到影格數")
    idxs = np.linspace(0, total - 1, SHEET_N).astype(int)
    got, _ = read_frames(args.video, idxs)

    cells = []
    for i in idxs:
        fr = got.get(int(i))
        if fr is None:
            continue
        c = third_panel(fr)
        c = cv2.resize(c, (260, int(260 * c.shape[0] / c.shape[1])))
        cv2.putText(c, str(int(i)), (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cells.append(c)
    if not cells:
        sys.exit("[錯誤] 取不到任何影格")

    h = min(c.shape[0] for c in cells)
    cells = [c[:h] for c in cells]
    ncol = 4
    rows = [cells[i:i + ncol] for i in range(0, len(cells), ncol)]
    for r in rows:
        while len(r) < ncol:
            r.append(np.full_like(cells[0], 255))
    grid = np.vstack([np.hstack(r) for r in rows])

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    p = od / "fig_4-14_candidates.png"
    cv2.imwrite(str(p), grid)
    print(f"[完成] {p}（共 {total} 影格，取樣 {len(cells)} 格）")
    print("\n僅顯示右側的純模型格，便於辨認姿勢。請挑三個**姿勢明確不同**者，"
          "\n並留意 4.6.2 節各項失敗情形所繫的姿勢："
          "\n  其一（原有服裝自肩部顯露）繫於**雙手上舉**，正文稱其為第一列。"
          "\n選定後執行："
          "\n  python make_fig414.py --video ... --rows a,b,c --out figures")


def main():
    ap = argparse.ArgumentParser(description="組出圖 4-14")
    ap.add_argument("--video", default="",
                    help="render_overlay.py 輸出的三格對照影片")
    ap.add_argument("--frames", default="",
                    help="三張已挑好的三格影像，逗號分隔（與 --video 擇一）")
    ap.add_argument("--rows", default="", help="三個影格序號，逗號分隔")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="掃過影片取得姿勢差異的經驗分布，供設定門檻")
    ap.add_argument("--calib-n", type=int, default=24,
                    help="--calibrate 取樣的影格數")
    ap.add_argument("--min-diff", type=float, default=0.0,
                    help="姿勢差異門檻（1 − IoU）；省略則用檔內的占位值")
    ap.add_argument("--allow-similar", action="store_true",
                    help="即使兩列姿勢相近仍出圖")
    args = ap.parse_args()

    import cv2

    if args.calibrate:
        if not args.video or not Path(args.video).exists():
            sys.exit("[錯誤] --calibrate 須以 --video 指定三格對照影片")
        calibrate(args)
        return

    if args.frames:
        paths = [Path(s.strip()) for s in args.frames.split(",") if s.strip()]
        if len(paths) != 3:
            sys.exit(f"[錯誤] --frames 須給三個檔案，收到 {len(paths)} 個")
        miss = [str(p) for p in paths if not p.exists()]
        if miss:
            sys.exit(f"[錯誤] 找不到：{'、'.join(miss)}")
        frames, idxs = [], []
        for p in paths:
            im = cv2.imread(str(p))
            if im is None:
                sys.exit(f"[錯誤] 讀不到 {p}")
            frames.append(im)
            idxs.append(p.stem)
        # 三張須同寬，否則下方的 (a)(b)(c) 欄位標示會對不準
        ws = {f.shape[1] for f in frames}
        if len(ws) > 1:
            print(f"[警告] 三張影像寬度不一（{sorted(ws)}），"
                  "將各自縮放至相同寬度；若其原本的三格切分比例不同，"
                  "欄位標示會錯位。")
    else:
        if not args.video:
            sys.exit("請以 --video 指定三格對照影片，或以 --frames 指定三張影像")
        if not Path(args.video).exists():
            sys.exit(f"[錯誤] 找不到 {args.video}。請先執行 render_overlay.py。")
        if args.sheet:
            sheet(args)
            return
        if not args.rows:
            sys.exit("請先執行 --sheet 挑選三個影格，再以 --rows 指定")

        idxs = [int(s) for s in args.rows.split(",") if s.strip()]
        if len(idxs) != 3:
            sys.exit(f"[錯誤] --rows 須給三個影格序號，收到 {len(idxs)} 個")

        got, total = read_frames(args.video, idxs)
        missing = [i for i in idxs if i not in got]
        if missing:
            sys.exit(f"[錯誤] 影片共 {total} 影格，取不到：{missing}")
        frames = [got[i] for i in idxs]

    # ---- 三列是否確實為不同姿勢 ----
    thr = args.min_diff if args.min_diff > 0 else POSE_DIFF_MIN
    print("\n各列之間的姿勢差異（右側純模型格之剪影，1 − IoU）：")
    ok = True
    for a, b in ((0, 1), (0, 2), (1, 2)):
        d = pose_diff(third_panel(frames[a]), third_panel(frames[b]))
        mark = "✓" if d >= thr else "✗"
        print(f"  {mark} 第 {a+1} 列（{idxs[a]}）對第 {b+1} 列"
              f"（{idxs[b]}）：{d:.4f}")
        if d < thr:
            ok = False

    if not (CALIBRATED or args.min_diff > 0):
        print(f"\n  [注意] 門檻 {thr} **尚未校準**，僅為占位值，"
              "據以判定「姿勢是否不同」並無依據。"
              "\n         請先執行 --calibrate 取得該影片上的經驗分布："
              "\n           python make_fig414.py --video <三格對照影片> --calibrate")

    if not ok:
        print(f"\n  ✗ 依門檻 {thr} 判定，有兩列的姿勢過於接近。"
              "\n     圖說稱「三列分別取自三個不同的姿勢」；若此判定成立，"
              "該敘述與事實不符，"
              "\n     且重複的列不提供新資訊卻佔去三分之一版面。"
              "\n     處理方式：先校準門檻確認判定可信，再改選影格；"
              "\n     或確認無誤後以 --allow-similar 出圖。")
        if not args.allow_similar:
            sys.exit(1)
    else:
        print("\n  ✓ 三列姿勢互異")

    # ---- 排版 ----
    rows = []
    for fr in frames:
        s = ROW_W / fr.shape[1]
        rows.append(cv2.resize(fr, (ROW_W, int(fr.shape[0] * s))))
    h = min(r.shape[0] for r in rows)
    rows = [r[:h] for r in rows]

    band = 40
    hdr = np.full((band, ROW_W, 3), 255, np.uint8)
    third = ROW_W // 3
    for i, t in enumerate(["(a) input frame",
                           "(b) garment composited",
                           "(c) full 3D avatar"]):
        cv2.putText(hdr, t, (i * third + 12, band - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)

    sep = np.full((6, ROW_W, 3), 255, np.uint8)
    fig = np.vstack([hdr] + [x for r in rows for x in (r, sep)][:-1])

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    p = od / "fig_4-14_compositing.png"
    cv2.imwrite(str(p), fig)
    print(f"\n[完成] {p}（影格 {idxs}）")
    print("\n核對：4.6.2 節其一所述「使用者左肩至腋下可見原有灰色上衣的殘留」"
          "\n須於第一列的 (b) 格可見；若該列非雙手上舉，正文的指涉須一併修正。")


if __name__ == "__main__":
    main()
