"""
assemble_repo.py

以「白名單」方式把論文相關檔案整理成一個乾淨的 GitHub repo 目錄。
只複製 repo_allowlist.txt 明確列出的檔案，其餘一律不碰，
並列出專案裡「沒被列入白名單」的檔案供你逐一檢視，避免漏掉該發布的東西。

用法（在 chupa 專案根目錄執行）：

    python assemble_repo.py --out ~/thesis-repo --dry-run   # 先看會複製什麼
    python assemble_repo.py --out ~/thesis-repo             # 實際複製
    python assemble_repo.py --out ~/thesis-repo --review    # 另列出未列入白名單的檔案

複製完成後，務必再跑 audit_repo.py 做推送前稽核。
"""
import argparse
import glob
import os
import shutil
import sys

ALLOWLIST = "repo_allowlist.txt"

# 這些副檔名即使被白名單誤列也不複製（第二道防線）
FORBIDDEN_EXT = {
    ".pkl", ".npz", ".npy", ".ckpt", ".pth", ".pt", ".safetensors", ".bin",
    ".obj", ".ply", ".glb", ".gltf", ".fbx", ".stl",
    ".pdf", ".mp4", ".mov", ".avi", ".mkv", ".zip", ".tar", ".gz", ".7z",
}

# --review 掃描時略過的目錄（第三方程式碼、權重、輸出）
SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "src", "scripts", "configs", "checkpoints", "data", "results",
    "demo_work", "debug_ablation", "debug_text_stage",
    "animation_data_emily_apose", "animation_data_apose",
    "animation_data_smpl", "animation_data_v2",
    "shape_assets", "shape_variants_driven", "outfit_change_figure",
    "img4640_driven_emily", "measurement_fit", "gender_beta_scan",
    "smpl2smplx_verify", "skinning_comparison", "m1_m2_apose_compare",
}


def read_allowlist(path):
    entries = []
    if not os.path.exists(path):
        sys.exit(f"[錯誤] 找不到白名單 {path}")
    for lineno, raw in enumerate(open(path, encoding="utf-8"), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            print(f"  [警告] 白名單第 {lineno} 行格式不符，略過：{raw.strip()}")
            continue
        entries.append((parts[0], parts[1]))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="輸出的 repo 目錄")
    ap.add_argument("--allowlist", default=ALLOWLIST)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--review", action="store_true",
                    help="另外列出專案根目錄中未列入白名單的 .py")
    args = ap.parse_args()

    out_root = os.path.expanduser(args.out)
    entries = read_allowlist(args.allowlist)

    copied, missing, blocked = [], [], []
    for dest_dir, pattern in entries:
        matches = sorted(glob.glob(pattern))
        if not matches:
            missing.append(pattern)
            continue
        for src in matches:
            if not os.path.isfile(src):
                continue
            ext = os.path.splitext(src)[1].lower()
            if ext in FORBIDDEN_EXT:
                blocked.append(src)
                continue
            dst = os.path.join(out_root, dest_dir, os.path.basename(src))
            copied.append((src, dst))
            if not args.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    print(f"{'[預演] ' if args.dry_run else ''}複製 {len(copied)} 個檔案 → {out_root}")
    for src, dst in copied:
        print(f"  {src}  ->  {os.path.relpath(dst, out_root)}")

    if blocked:
        print(f"\n[攔截] 下列檔案副檔名在禁止清單中，未複製（請確認白名單是否寫錯）：")
        for p in blocked:
            print(f"  {p}")

    if missing:
        print(f"\n[找不到] 白名單列了但專案中不存在（檔名可能不同，請修正白名單）：")
        for p in missing:
            print(f"  {p}")

    if args.review:
        listed = {os.path.basename(s) for s, _ in copied}
        extras = [f for f in sorted(glob.glob("*.py"))
                  if os.path.basename(f) not in listed]
        print(f"\n[待檢視] 專案根目錄中未列入白名單的 .py（{len(extras)} 個）：")
        print("         逐一判斷是你寫的（加進白名單）還是第三方的（維持排除）")
        for f in extras:
            print(f"  {f}")

    if not args.dry_run and copied:
        print(f"\n下一步：")
        print(f"  cd {out_root}")
        print(f"  python audit_repo.py .        # 推送前稽核")


if __name__ == "__main__":
    main()
