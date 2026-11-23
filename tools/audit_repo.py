"""
audit_repo.py

GitHub 推送前的稽核：掃描 repo 目錄，找出可能有著作權或隱私疑慮的檔案。
每次 push 前都跑一次。

用法：

    python audit_repo.py ~/thesis-repo
    python audit_repo.py ~/thesis-repo --staged    # 只檢查 git 已追蹤/暫存的檔案

結束碼 0 表示通過，1 表示有需要處理的項目。
"""
import argparse
import os
import re
import subprocess
import sys

MB = 1024 * 1024

# (副檔名, 風險說明)
RISKY_EXT = {
    ".pkl": "模型參數或資料集擬合結果（SMPL-X / RenderPeople 授權禁止散布）",
    ".npz": "模型參數或資料集擬合結果",
    ".npy": "權重或頂點陣列，可能為衍生資料",
    ".ckpt": "模型權重",
    ".pth": "模型權重",
    ".pt": "模型權重",
    ".safetensors": "模型權重",
    ".bin": "模型權重或二進位資料",
    ".obj": "生成網格（由第三方模型與資料集衍生）",
    ".ply": "生成網格",
    ".glb": "生成網格（含貼圖）",
    ".gltf": "生成網格",
    ".fbx": "生成網格",
    ".pdf": "論文 PDF（出版社著作權）",
    ".mp4": "影片（素材授權與人物肖像權）",
    ".mov": "影片（素材授權與人物肖像權）",
    ".avi": "影片",
    ".mkv": "影片",
    ".exe": "執行檔（第三方軟體不可散布）",
    ".dll": "第三方二進位檔",
    ".so": "第三方二進位檔",
    ".zip": "壓縮檔，內容不明",
    ".tar": "壓縮檔，內容不明",
    ".gz": "壓縮檔，內容不明",
}

# 檔名或路徑中出現這些字樣就提醒
RISKY_NAME = [
    (r"renderpeople", "RenderPeople 商業資料集"),
    (r"smplx?_(neutral|male|female)", "SMPL/SMPL-X 模型檔"),
    (r"thuman", "THuman 資料集"),
    (r"checkpoint", "模型權重目錄"),
    (r"\bchupa\.py\b", "Chupa 原始碼（無授權，不可散布）"),
    (r"stable[-_]?diffusion", "Stable Diffusion 權重"),
    (r"controlnet", "ControlNet 權重"),
]

# 原始碼中若出現這些，可能夾帶了第三方程式碼或金鑰
RISKY_CONTENT = [
    (r"(?i)copyright\s+\(c\)\s+\d{4}.*(max[- ]planck|renderpeople|snuvclab)",
     "疑似第三方版權標頭"),
    (r"(?i)(api[-_]?key|secret[-_]?key|password)\s*=\s*['\"][^'\"]{8,}", "疑似硬編碼金鑰"),
    (r"ghp_[A-Za-z0-9]{30,}", "疑似 GitHub token"),
]

TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".cfg", ".ini", ".sh"}


def git_files(root):
    try:
        out = subprocess.run(["git", "-C", root, "ls-files"],
                             capture_output=True, text=True, check=True).stdout
        return [os.path.join(root, p) for p in out.splitlines() if p]
    except Exception as e:
        sys.exit(f"[錯誤] 無法取得 git 檔案清單：{e}")


def walk_files(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            found.append(os.path.join(dirpath, fn))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--staged", action="store_true", help="只檢查 git 追蹤中的檔案")
    ap.add_argument("--max-mb", type=float, default=50.0, help="超過此大小即警告")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        sys.exit(f"[錯誤] 找不到目錄 {root}")

    files = git_files(root) if args.staged else walk_files(root)
    print(f"稽核 {len(files)} 個檔案於 {root}\n")

    blockers, warnings = [], []

    for path in sorted(files):
        rel = os.path.relpath(path, root)
        low = rel.lower()
        ext = os.path.splitext(rel)[1].lower()

        if ext in RISKY_EXT:
            blockers.append((rel, f"副檔名 {ext}：{RISKY_EXT[ext]}"))

        for pat, why in RISKY_NAME:
            if re.search(pat, low):
                blockers.append((rel, f"檔名/路徑含敏感字樣：{why}"))

        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > 100 * MB:
            blockers.append((rel, f"{size/MB:.0f} MB，超過 GitHub 單檔 100 MB 上限"))
        elif size > args.max_mb * MB:
            warnings.append((rel, f"{size/MB:.0f} MB，偏大，考慮改放 Releases 或外部連結"))

        if ext in TEXT_EXT and size < 2 * MB:
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat, why in RISKY_CONTENT:
                if re.search(pat, text):
                    blockers.append((rel, why))

    # 去重
    seen = set()
    blockers = [b for b in blockers if not (b in seen or seen.add(b))]

    if blockers:
        print(f"[需處理] {len(blockers)} 項：")
        for rel, why in blockers:
            print(f"  ✗ {rel}\n      {why}")
        print()
    if warnings:
        print(f"[提醒] {len(warnings)} 項：")
        for rel, why in warnings:
            print(f"  ! {rel}\n      {why}")
        print()

    if not blockers and not warnings:
        print("未發現可疑檔案。")

    print("自動稽核無法涵蓋的部分，請自行確認：")
    print("  - 影片與照片中可辨識的人物是否已取得同意")
    print("  - 文件中引用的圖表是否為你自己產生")
    print("  - README 是否已說明第三方素材的取得方式而非直接附上")

    sys.exit(1 if blockers else 0)


if __name__ == "__main__":
    main()
