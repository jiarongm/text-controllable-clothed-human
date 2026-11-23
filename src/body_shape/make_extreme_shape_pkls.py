"""
make_extreme_shape_pkls.py

為「體型差異對照」建立三組極端體型的 subject pkl：
  155cm/75kg、165cm/60kg、180cm/58kg
在 chupa 環境、chupa 專案目錄執行。
"""
import json
import pickle
from pathlib import Path

import numpy as np

from fit_betas_from_measurements import fit

TEMPLATE_PKL = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
OUT_DIR = TEMPLATE_PKL.parent

SHAPES = [(155, 75), (165, 60), (180, 58)]
BETA_WARN = 2.0   # 超過此值，服裝生成品質可能下降

with open(TEMPLATE_PKL, "rb") as f:
    template = pickle.load(f, encoding="latin1")
orig = np.asarray(template["betas"])

index = []
for h, w in SHAPES:
    name = f"rp_ext_{h}_{w}_001"
    print(f"\n=== {name}: {h}cm / {w}kg (BMI {w / (h / 100) ** 2:.1f}) ===")
    betas, _, _ = fit(h, w)

    max_abs = float(np.abs(betas).max())
    flag = "  <-- 偏大，留意服裝品質" if max_abs > BETA_WARN else ""
    print(f"  betas 絕對值上限: {max_abs:.2f}{flag}")

    data = dict(template)
    data["betas"] = betas.astype(orig.dtype).reshape(orig.shape)
    with open(OUT_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"  已建立 {OUT_DIR / (name + '.pkl')}")
    index.append({"subject": name, "height_cm": h, "weight_kg": w})

with open("extreme_shape_index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)
print("\n[DONE] 體型清單已存至 extreme_shape_index.json")
