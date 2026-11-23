"""
make_subject_from_measurements.py

把「身高體重擬合出的 betas」做成 Chupa 可用的 subject pkl。
可一次建立多組體型，供生成不同身形的著衣網格。

在 chupa 環境、chupa 專案目錄執行。

用法：
  1. 先跑 fit_betas_from_measurements.py 產生 measurement_fit/*_betas.npy
  2. 再跑本腳本，把那些 betas 轉成 subject pkl
"""
import pickle
import numpy as np
from pathlib import Path

# 模板：沿用 emily A-pose 的 pkl（取其 A-pose 姿態、body_scale、translation）
TEMPLATE_PKL = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
BETAS_DIR = Path("measurement_fit")
OUT_DIR = TEMPLATE_PKL.parent

# 要建立的 subject：{subject名稱: betas檔名}
SUBJECTS = {
    "rp_meas_tall_thin_001":   "tall_thin_betas.npy",
    "rp_meas_average_001":     "average_betas.npy",
    "rp_meas_short_heavy_001": "short_heavy_betas.npy",
}

with open(TEMPLATE_PKL, "rb") as f:
    template = pickle.load(f, encoding="latin1")

orig_betas = np.asarray(template["betas"])
print(f"[INFO] 模板 betas shape: {orig_betas.shape}")

for subject, betas_file in SUBJECTS.items():
    betas_path = BETAS_DIR / betas_file
    if not betas_path.exists():
        print(f"[跳過] {subject}: 找不到 {betas_path}")
        continue

    betas = np.load(betas_path).reshape(-1)[:10]

    data = dict(template)
    data["betas"] = betas.astype(orig_betas.dtype).reshape(orig_betas.shape)

    out_path = OUT_DIR / f"{subject}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  已建立 {out_path}")
    print(f"    betas: {betas.round(3)}")

print("\n[DONE] subject pkl 已建立。生成指令範例：")
print("""
for subj in rp_meas_tall_thin_001 rp_meas_average_001 rp_meas_short_heavy_001; do
  python scripts/chupa.py configs/text.yaml \\
    dataset.subject=$subj \\
    'chupa.prompt=a woman wearing a long dress' \\
    chupa.seed=42 \\
    chupa.text_steps=100 chupa.text_cfg_scale=7.5 chupa.text_image_cfg_scale=1.5 \\
    chupa.n_resample_from_text=0 chupa.n_resample=0
done
""")
