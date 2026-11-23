"""
batch_text_ablation.py

批次跑多組 prompt × 多個 seed 的文字生成，只跑到第一階段（stage1）法向量圖即停止，
用於大量評估「修正後文字對服裝類型的控制力」（論文 4.4.3 節）。

原理：服裝類型於文字擴散模型的第一階段即決定（消融已證實停用 resample 後保留至最終），
故只需跑 diffusion 的第一階段（約 30-40 秒），不必等數十分鐘的網格雕刻。

前置需求：
  scripts/chupa.py 的 difftext_sample() 已加入 stage1 存檔（存到 debug_ablation/），
  且已可從命令列以 chupa.n_resample_from_text=0 等參數覆寫（本專案先前已完成）。

用法：
  conda activate chupa
  cd ~/projects/chupa
  python batch_text_ablation.py

輸出：
  debug_ablation/ 底下的 stage1_*.png（檔名含 prompt 與 seed）
  batch_ablation_index.csv（提示、種子、類別、檔名對照，供人工判定填寫正確與否）
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# 測試提示：依類別組織。可自行增減，建議每類別 3-5 組、總數 12-20 組。
# ============================================================
PROMPTS = {
    "上衣_袖長": [
        "a person wearing a t-shirt",
        "a person wearing a long-sleeve shirt",
        "a person wearing a tank top",
        "a person wearing a hooded sweatshirt",
    ],
    "褲裝_長短": [
        "a person wearing shorts",
        "a person wearing long pants",
        "a person wearing jeans",
    ],
    "裙裝_連身": [
        "a woman wearing a long dress",
        "a woman wearing a short dress",
        "a woman wearing a summer dress",
    ],
    "寬鬆_組合": [
        "a person wearing a long coat",
        "a woman wearing a skirt",
        "a person wearing a puffer jacket",
        "a woman wearing a blouse and a skirt",
    ],
}

SEEDS = [42, 7, 123]          # 每組提示使用的隨機種子
SUBJECT = "rp_emily_apose_001"

# 修正後之固定設定
FIXED_ARGS = [
    "chupa.n_resample_from_text=0",
    "chupa.text_steps=100",
    "chupa.text_cfg_scale=7.5",
    "chupa.text_image_cfg_scale=1.5",
]

# stage1 存檔位置（difftext_sample 內設定的資料夾）
ABLATION_DIR = Path("debug_ablation")


def stage1_filename(prompt, seed):
    """對應 difftext_sample 內 stage1 的檔名規則。
    注意：若你的 difftext_sample 檔名規則與此不同，請據實修改此函式。"""
    safe = "_".join(str(prompt).split())
    safe = "".join(c for c in safe if c.isalnum() or c in "_-")[:60]
    # 專案先前的規則為 stage1_text_{prompt}_seed{seed}_cfg{cfg}_nres{n}.png
    # 這裡以 glob 方式尋找，避免 cfg 浮點格式差異
    return f"stage1_text_{safe}_seed{seed}_"


def run_one(prompt, seed, timeout_sec=180):
    """跑一次生成，只等到 stage1 存檔出現就中止（省時）。"""
    ABLATION_DIR.mkdir(exist_ok=True)
    prefix = stage1_filename(prompt, seed)

    # 記錄跑之前既有的 stage1 檔，用來偵測新檔
    before = set(ABLATION_DIR.glob(f"{prefix}*.png"))

    cmd = [
        sys.executable, "scripts/chupa.py", "configs/text.yaml",
        f"dataset.subject={SUBJECT}",
        f"chupa.prompt={prompt}",
        f"chupa.seed={seed}",
    ] + FIXED_ARGS

    print(f"\n[BATCH] prompt='{prompt}' seed={seed}")
    print("        ", " ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 輪詢等待 stage1 檔出現
    t0 = time.time()
    found = None
    while time.time() - t0 < timeout_sec:
        now = set(ABLATION_DIR.glob(f"{prefix}*.png"))
        new = now - before
        if new:
            found = sorted(new)[0]
            break
        if proc.poll() is not None:
            # 行程已結束仍沒找到 stage1，可能檔名規則不符
            break
        time.sleep(2)

    # stage1 已存檔，中止後續昂貴的雕刻
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    if found:
        print(f"        stage1 -> {found.name}  (耗時 {time.time()-t0:.0f}s)")
    else:
        print(f"        [警告] 未偵測到 stage1 檔，請確認 difftext_sample 的檔名規則與 stage1_filename() 一致")
    return found


def main():
    rows = []
    total = sum(len(v) for v in PROMPTS.values()) * len(SEEDS)
    print(f"[BATCH] 共 {total} 次生成（每次只跑到 stage1，約 30-60 秒）")

    for category, prompts in PROMPTS.items():
        for prompt in prompts:
            for seed in SEEDS:
                fpath = run_one(prompt, seed)
                rows.append({
                    "category": category,
                    "prompt": prompt,
                    "seed": seed,
                    "stage1_file": fpath.name if fpath else "MISSING",
                    "type_correct": "",   # ← 供你事後人工填寫 1/0
                })

    # 輸出對照表供人工判定
    out_csv = Path("batch_ablation_index.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category", "prompt", "seed", "stage1_file", "type_correct"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[DONE] 對照表已輸出：{out_csv}")
    print("       請開啟 debug_ablation/ 逐張判定服裝類型是否正確，")
    print("       於 batch_ablation_index.csv 的 type_correct 欄填 1（正確）或 0（錯誤），")
    print("       再回傳給我，我幫你算各類別正確率並填入論文表 4-6。")


if __name__ == "__main__":
    main()

