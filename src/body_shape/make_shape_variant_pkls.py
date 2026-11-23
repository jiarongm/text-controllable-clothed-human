"""
make_shape_variant_pkls.py

以 emily A-pose 的 pkl 為模板，製作三個不同身形的 subject pkl，
供「同一件衣服、不同身形」的生成 demo 使用。
在 chupa 環境執行。

產出：
  data/renderpeople/smplx/rp_shape_tall_001.pkl    高瘦
  data/renderpeople/smplx/rp_shape_avg_001.pkl     標準
  data/renderpeople/smplx/rp_shape_heavy_001.pkl   矮壯
"""
import pickle
import numpy as np
import shutil
from pathlib import Path

TEMPLATE_PKL = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
OUT_DIR = TEMPLATE_PKL.parent

# 三種身形的 betas 設定（beta[0]=身高、beta[1]=胖瘦；方向已驗證）
# 注意：beta[1] 的胖瘦方向依你 Blender 觀察為準，若方向相反，把正負號對調
SHAPES = {
    "rp_shape_tall_001":  [ 2.0, -1.5, 0,0,0,0,0,0,0,0],   # 高、瘦
    "rp_shape_avg_001":   [ 0.0,  0.0, 0,0,0,0,0,0,0,0],   # 標準
    "rp_shape_heavy_001": [-1.5,  1.8, 0,0,0,0,0,0,0,0],   # 矮、壯
}

with open(TEMPLATE_PKL, "rb") as f:
    template = pickle.load(f, encoding="latin1")

print(f"[INFO] 模板欄位: {list(template.keys())}")
print(f"[INFO] 模板 betas: {np.asarray(template['betas']).reshape(-1).round(3)}")

for name, betas in SHAPES.items():
    data = dict(template)  # 淺拷貝即可，我們只換 betas
    orig = np.asarray(template["betas"])
    new_betas = np.array(betas, dtype=orig.dtype).reshape(orig.shape)
    data["betas"] = new_betas

    out_path = OUT_DIR / f"{name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  已建立 {out_path}  betas={betas[:2]}...")

print("\n[DONE] 三個身形 pkl 已建立。")
print("注意：Chupa 的 dataset 可能要求 subject 在特定清單或資料夾結構中；")
print("若生成時報找不到 subject，請確認 data/renderpeople/ 下是否需要對應的其他檔案。")
