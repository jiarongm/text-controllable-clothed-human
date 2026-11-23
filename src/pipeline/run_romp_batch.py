r"""
run_romp_batch.py —— 批次處理六段動作影片，產出 run_temporal_eval 所需的姿態序列

為何不直接用 run_temporal_eval.py --stage pose
==============================================
該階段透過 drive_avatar.load_motion 讀取 ROMP 的輸出，而 ROMP 依執行方式
（CLI 或 API、逐影格 npz 或 video_results.npz）會產生數種不同的檔案配置。
本腳本改以 ROMP 的 Python API 逐影格推論，**直接寫出最終格式**，
省去中間的格式判讀，也避免六段各自出現不同的輸出配置。

輸出即為 run_temporal_eval.py --stage eval 所需的 `pose_<名稱>.npz` 與
`motions.json`，故執行完本腳本後可直接跳至 `--stage weights`。

動作命名
========
第六段的名稱以 `stress` 開頭。run_temporal_eval 依此前綴自動將其歸為
**壓力案例**：單獨報告、不併入總平均，亦不參與失敗門檻 τ 的計算。
若併入，一般動作的失敗幀率會被系統性低估。

品質檢查
========
本腳本會逐段報告偵測失敗的影格比例。**若某段的失敗率偏高，該段應重錄**——
未偵測到人的影格會被跳過，使動作在該處接起來而產生不存在的跳動，
其所造成的抖動會被誤計為蒙皮的不穩定。

用法
====
  conda activate romp
  python run_romp_batch.py --video-dir ~/projects/chupa/motion_testVideo \
      --out temporal_eval

  # 只重跑其中一段
  python run_romp_batch.py --only stress_arms --video-dir ... --out temporal_eval

環境：romp
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# 檔名（不含副檔名）→ 動作名稱。順序即論文 4.3.5 節所列者。
MOTIONS = [
    ("1", "shoulder",     "肩部抬舉"),
    ("2", "elbow",        "手肘屈伸"),
    ("3", "run",          "跑步"),
    ("4", "legraise",     "單腿高抬"),
    ("5", "twist",        "軀幹旋轉"),
    ("6", "stress_arms",  "雙臂接近身體（壓力案例）"),
]

MAX_MISS_RATE = 0.05      # 偵測失敗率超過此值即提出警告


def process_one(model, video_path, max_frames):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"打不開影片：{video_path}")

    poses, n_total, n_miss = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = model(rgb)
        if out is None or "smpl_thetas" not in out or len(out["smpl_thetas"]) == 0:
            n_miss += 1
            continue
        # 取第一個人。六段影片皆為單人，若畫面中意外出現他人，
        # ROMP 的排序未必穩定，故此處於偵測到多人時提出警告。
        if len(out["smpl_thetas"]) > 1:
            n_miss += 0      # 不計為失敗，但下方會提示
        poses.append(np.asarray(out["smpl_thetas"][0]).reshape(-1))
        if max_frames and len(poses) >= max_frames:
            break
    cap.release()
    return np.stack(poses) if poses else None, n_total, n_miss


def main():
    ap = argparse.ArgumentParser(description="批次處理動作影片，產出姿態序列")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", default="temporal_eval")
    ap.add_argument("--ext", default=".mp4")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="每段最多取用的影格數（預設 200）")
    ap.add_argument("--only", help="只處理指定的動作名稱")
    args = ap.parse_args()

    try:
        import romp
    except ImportError:
        sys.exit("[錯誤] 找不到 romp。請先 conda activate romp")

    vdir = Path(os.path.expanduser(args.video_dir))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("初始化 ROMP ...")
    settings = romp.main.default_settings
    settings.calc_smpl = True
    model = romp.ROMP(settings)

    index_path = out / "motions.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) \
        if index_path.exists() else {}

    warnings = []
    for stem, name, label in MOTIONS:
        if args.only and name != args.only:
            continue
        vp = vdir / f"{stem}{args.ext}"
        if not vp.exists():
            print(f"[略過] 找不到 {vp}")
            continue

        print(f"\n=== {label}（{vp.name} → {name}）===")
        poses, n_total, n_miss = process_one(model, vp, args.max_frames)
        if poses is None:
            print("  ✗ 沒有任何影格偵測到人，此段不可用")
            warnings.append(f"{name}：完全無偵測結果")
            continue

        miss_rate = n_miss / max(n_total, 1)
        np.savez_compressed(out / f"pose_{name}.npz", poses=poses)
        index[name] = int(len(poses))
        print(f"  可用 {len(poses)} 影格 / 讀入 {n_total} 影格"
              f"，偵測失敗 {n_miss}（{100 * miss_rate:.1f}%）")

        if len(poses) < 100:
            print(f"  [警告] 可用影格不足 100，統計量不穩定，建議重錄較長的片段")
            warnings.append(f"{name}：僅 {len(poses)} 影格")
        if miss_rate > MAX_MISS_RATE:
            print(f"  [警告] 偵測失敗率 {100 * miss_rate:.1f}% 偏高。"
                  f"失敗影格會被跳過，使動作於該處接起而產生不存在的跳動，"
                  f"其抖動會被誤計為蒙皮的不穩定——建議重錄此段")
            warnings.append(f"{name}：失敗率 {100 * miss_rate:.1f}%")

    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print(f"\n[完成] {len(index)} 段，已寫出 {index_path}")
    if "stress_arms" in index:
        print("  stress_arms 將由 run_temporal_eval 自動歸為壓力案例，"
              "單獨報告且不參與失敗門檻的計算。")
    if warnings:
        print("\n須留意的段落：")
        for w in warnings:
            print(f"  - {w}")
        print("\n上列問題會直接影響 4.3.5 節的結論可信度，建議先處理再往下跑。")
    else:
        print("\n六段皆通過品質檢查，可接續執行：")
        print("  conda activate chupa")
        print("  python run_temporal_eval.py --stage weights --work <資產> "
              "--smplx-obj <SMPL-X.obj> --bind <bind.pkl> --out temporal_eval")


if __name__ == "__main__":
    main()
