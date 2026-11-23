"""
batch_texture_cases.py

對九組案例批次執行貼圖流程（build_asset.py 五步一次跑完）。
會先確認各 subject 的 debug_source_smplx_<subject>.obj 存在，缺的自動產生。

在 chupa 環境、chupa 專案目錄執行（先關閉 Gradio，確認幾何已生成完）：

    python batch_texture_cases.py --check          # 只檢查前置條件，不執行
    nohup python batch_texture_cases.py > texture_cases_gen.log 2>&1 &
    python batch_texture_cases.py --only 4         # 單組重跑

輸出：
    demo_work/case<N>_<短名>/     texture.png、chupa_textured.glb、label_preview.png 等
    texture_cases_index.csv       每組解析到的顏色、覆蓋率、耗時、狀態
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

# (編號, 身高, 體重, 短名, prompt 中用於比對 mesh 的獨特片段, 完整 prompt)
CASES = [
    (1, 155, 46, "white_tshirt",   "white_t-shirt",     "a woman wearing a white t-shirt and blue jeans"),
    (2, 155, 53, "red_hoodie",     "red_hoodie",        "a woman wearing a red hoodie and black long pants"),
    (3, 155, 65, "yellow_tank",    "yellow_tank_top",   "a woman wearing a yellow tank top and gray shorts"),
    (4, 165, 52, "blue_blazer",    "blue_blazer",       "a woman wearing a blue blazer and white long pants"),
    (5, 165, 60, "green_dress",    "green_long_dress",  "a woman wearing a green long dress"),
    (6, 165, 74, "brown_coat",     "brown_long_coat",   "a woman wearing a brown long coat and black long pants"),
    (7, 175, 58, "pink_blouse",    "pink_blouse",       "a woman wearing a pink blouse and a black skirt"),
    (8, 175, 67, "purple_shirt",   "purple_long-sleeve", "a woman wearing a purple long-sleeve shirt and gray long pants"),
    (9, 175, 83, "orange_puffer",  "orange_puffer",     "a woman wearing an orange puffer jacket and blue jeans"),
]

INDEX_CSV = "texture_cases_index.csv"
FIELDS = ["no", "subject", "kind_short", "prompt", "work", "upper_color",
          "lower_color", "coverage", "minutes", "status"]


def subject_of(h, w):
    return f"rp_lib_{h}_{w}_001"


def mesh_for(subject, frag):
    """找該 subject 底下、資料夾名含 frag 的 mesh_refine.obj。"""
    pat = os.path.join("results", "renderpeople", "*", subject, f"*{frag}*", "mesh_refine.obj")
    cands = glob.glob(pat)
    return max(cands, key=os.path.getmtime) if cands else ""


def ensure_source_obj(subject):
    """semantic 步驟需要 debug_source_smplx_<subject>.obj，缺的話產生。"""
    path = f"debug_source_smplx_{subject}.obj"
    if os.path.exists(path):
        return path, "exists"
    print(f"  [前置] 產生 {path}", flush=True)
    r = subprocess.run([sys.executable, "export_source_smplx.py", "--subject", subject])
    return (path, "created") if (r.returncode == 0 and os.path.exists(path)) else ("", "failed")


def run_case(no, h, w, short, frag, prompt):
    subject = subject_of(h, w)
    work = os.path.join("demo_work", f"case{no}_{short}")
    row = dict(no=no, subject=subject, kind_short=short, prompt=prompt, work=work,
               upper_color="", lower_color="", coverage="", minutes="", status="")

    if not mesh_for(subject, frag):
        row["status"] = "NO_MESH（幾何尚未生成或片段比對不到）"
        return row
    src, src_state = ensure_source_obj(subject)
    if src_state == "failed":
        row["status"] = "SOURCE_OBJ_FAILED"
        return row

    env = dict(os.environ)
    env.pop("UVBAKE_FAST", None)          # 確保走 50 視角的品質組態

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "build_asset.py", "--match", frag, "--work", work],
        env=env, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-3000:], flush=True)        # 保留尾段到 log，方便事後追查

    m = re.search(r"upper\s*=\s*([^,\s]+)\s*,\s*lower\s*=\s*(\S+)", out)
    if m:
        row["upper_color"], row["lower_color"] = m.group(1), m.group(2)
    m = re.search(r"覆蓋率[^:：]*[:：]\s*([\d.]+)\s*%", out)
    if m:
        row["coverage"] = m.group(1)

    row["minutes"] = f"{(time.time() - t0) / 60:.1f}"
    glb = os.path.join(work, "chupa_textured.glb")
    row["status"] = "OK" if (proc.returncode == 0 and os.path.exists(glb)) \
        else f"FAILED(returncode={proc.returncode})"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="只檢查前置條件")
    args = ap.parse_args()

    cases = [c for c in CASES if args.only is None or c[0] == args.only]

    if args.check:
        print(f"{'#':>2}  {'subject':<20} {'mesh':<6} {'source obj':<10}  prompt")
        for no, h, w, short, frag, prompt in cases:
            s = subject_of(h, w)
            print(f"{no:>2}  {s:<20} {'OK' if mesh_for(s, frag) else '缺':<6} "
                  f"{'OK' if os.path.exists(f'debug_source_smplx_{s}.obj') else '待產生':<10}  {prompt}")
        return

    rows = []
    for no, h, w, short, frag, prompt in cases:
        print(f"\n===== [{no}/9] {subject_of(h, w)} | {prompt} | {time.strftime('%H:%M:%S')} =====",
              flush=True)
        row = run_case(no, h, w, short, frag, prompt)
        print(f"===== [{no}/9] {row['status']}｜顏色 upper={row['upper_color'] or '?'} "
              f"lower={row['lower_color'] or '?'}｜覆蓋率 {row['coverage'] or '?'}%"
              f"｜{row['minutes'] or '-'} 分鐘 =====", flush=True)
        rows.append(row)

        mode = "a" if (args.only is not None and os.path.exists(INDEX_CSV)) else "w"
        with open(INDEX_CSV, mode, newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            if mode == "w":
                wr.writeheader()
                wr.writerows(rows)
            else:
                wr.writerow(row)

    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"\n[完成] {ok}/{len(rows)} 組成功，紀錄於 {INDEX_CSV}")
    low = [r["no"] for r in rows if r["coverage"] and float(r["coverage"]) < 95]
    if low:
        print(f"[注意] 覆蓋率低於 95% 的組別: {low}（檢查 UVBAKE_FAST 是否已 unset）")


if __name__ == "__main__":
    main()
