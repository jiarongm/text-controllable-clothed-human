"""
batch_outfit_cases.py

九組「體型 × 顏色 × 服裝類型」案例的批次幾何生成。
沿用體型庫已建立的 subject pkl（rp_lib_*），不需重新擬合體型。

在 chupa 環境、chupa 專案目錄執行（先關閉 Gradio）：

    nohup python batch_outfit_cases.py > outfit_cases_gen.log 2>&1 &

    python batch_outfit_cases.py --only 4      # 只跑第 4 組（測試或重跑用）
    python batch_outfit_cases.py --list        # 只列出案例，不生成

輸出：
    outfit_cases_index.csv   每組的 subject、prompt、mesh 路徑、耗時
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

# (編號, 身高, 體重, 服裝類型, prompt)
CASES = [
    (1, 155, 46, "T 恤＋牛仔褲",     "a woman wearing a white t-shirt and blue jeans"),
    (2, 155, 53, "連帽衫＋長褲",     "a woman wearing a red hoodie and black long pants"),
    (3, 155, 65, "背心＋短褲",       "a woman wearing a yellow tank top and gray shorts"),
    (4, 165, 52, "西裝外套＋長褲",   "a woman wearing a blue blazer and white long pants"),
    (5, 165, 60, "連身長裙",         "a woman wearing a green long dress"),
    (6, 165, 74, "長大衣＋長褲",     "a woman wearing a brown long coat and black long pants"),
    (7, 175, 58, "女用襯衫＋裙",     "a woman wearing a pink blouse and a black skirt"),
    (8, 175, 67, "長袖襯衫＋長褲",   "a woman wearing a purple long-sleeve shirt and gray long pants"),
    (9, 175, 83, "羽絨外套＋牛仔褲", "a woman wearing an orange puffer jacket and blue jeans"),
]

FIXED_ARGS = [
    "chupa.seed=42",
    "chupa.text_steps=100",
    "chupa.text_cfg_scale=7.5",
    "chupa.text_image_cfg_scale=1.5",
    "chupa.n_resample_from_text=0",
    "chupa.n_resample=0",
]
INDEX_CSV = "outfit_cases_index.csv"


def subject_of(h, w):
    return f"rp_lib_{h}_{w}_001"


def find_new_mesh(subject, since):
    pat = os.path.join("results", "renderpeople", "*", subject, "*", "mesh_refine.obj")
    cands = [c for c in glob.glob(pat) if os.path.getmtime(c) >= since]
    return max(cands, key=os.path.getmtime) if cands else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="只跑指定編號")
    ap.add_argument("--list", action="store_true", help="只列出案例")
    args = ap.parse_args()

    cases = [c for c in CASES if args.only is None or c[0] == args.only]

    # 先檢查 subject pkl 是否都存在，避免跑到一半才失敗
    missing = [subject_of(h, w) for _, h, w, _, _ in cases
               if not os.path.exists(f"data/renderpeople/smplx/{subject_of(h, w)}.pkl")]
    for no, h, w, kind, prompt in cases:
        print(f"  [{no}] {h}cm/{w}kg  {kind}  |  {prompt}")
    if missing:
        sys.exit(f"[錯誤] 缺少 subject pkl: {missing}")
    if args.list:
        return

    rows = []
    for no, h, w, kind, prompt in cases:
        subject = subject_of(h, w)
        print(f"\n===== [{no}/9] 開始 {subject} | {prompt} | {time.strftime('%H:%M:%S')} =====",
              flush=True)
        t0 = time.time()
        cmd = [sys.executable, "scripts/chupa.py", "configs/text.yaml",
               f"dataset.subject={subject}", f"chupa.prompt={prompt}"] + FIXED_ARGS
        ret = subprocess.run(cmd).returncode
        mesh = find_new_mesh(subject, t0) if ret == 0 else ""
        minutes = (time.time() - t0) / 60
        status = "OK" if mesh else f"FAILED(returncode={ret})"
        print(f"===== [{no}/9] {status}，耗時 {minutes:.1f} 分鐘 =====", flush=True)
        rows.append(dict(no=no, height_cm=h, weight_kg=w, subject=subject, kind=kind,
                         prompt=prompt, mesh=mesh, minutes=f"{minutes:.1f}", status=status))

        # 每跑完一組就寫一次，中斷也不會遺失紀錄
        mode = "a" if (args.only is not None and os.path.exists(INDEX_CSV)) else "w"
        with open(INDEX_CSV, mode, newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if mode == "w":
                wr.writeheader()
                wr.writerows(rows)
            else:
                wr.writerow(rows[-1])

    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n[完成] {ok}/{len(rows)} 組成功，紀錄於 {INDEX_CSV}")


if __name__ == "__main__":
    main()
