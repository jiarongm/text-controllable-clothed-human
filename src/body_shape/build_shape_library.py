"""
build_shape_library.py
建立體型庫：依身高 × BMI 組合，自動擬合 betas 並建立 subject pkl。
在 chupa 環境執行。
"""
import pickle
import numpy as np
from pathlib import Path
from fit_betas_from_measurements import fit

TEMPLATE_PKL = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
OUT_DIR = TEMPLATE_PKL.parent

# 身高 × BMI 網格
HEIGHTS = [155, 165, 175]
BMIS = {"thin": 19, "normal": 22, "heavy": 27}

with open(TEMPLATE_PKL, "rb") as f:
    template = pickle.load(f, encoding="latin1")
orig_betas = np.asarray(template["betas"])

subjects = []
for h in HEIGHTS:
    for bmi_name, bmi in BMIS.items():
        w = round(bmi * (h / 100.0) ** 2)
        name = f"rp_lib_{h}_{w}_001"
        print(f"\n=== {name}: {h}cm / {w}kg (BMI {bmi}) ===")

        betas, _, _ = fit(h, w)

        data = dict(template)
        data["betas"] = betas.astype(orig_betas.dtype).reshape(orig_betas.shape)
        out_path = OUT_DIR / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  已建立 {out_path}")
        subjects.append((name, h, w))

# 輸出生成指令
print("\n\n=== 生成指令（可貼進終端機掛著跑）===")
print("for subj in " + " ".join(s[0] for s in subjects) + "; do")
print("""  python scripts/chupa.py configs/text.yaml \\
    dataset.subject=$subj \\
    'chupa.prompt=a woman wearing a long dress' \\
    chupa.seed=42 \\
    chupa.text_steps=100 chupa.text_cfg_scale=7.5 chupa.text_image_cfg_scale=1.5 \\
    chupa.n_resample_from_text=0 chupa.n_resample=0
done""")

# 存成 json 供之後 build_shape_assets 使用
import json
with open("shape_library_index.json", "w", encoding="utf-8") as f:
    json.dump([{"subject": s[0], "height_cm": s[1], "weight_kg": s[2]} for s in subjects],
              f, indent=2, ensure_ascii=False)
print("\n體型清單已存至 shape_library_index.json")
