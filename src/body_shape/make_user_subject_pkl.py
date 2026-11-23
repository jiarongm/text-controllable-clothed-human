"""
make_user_subject_pkl.py
把 transfer_model 轉換出的 SMPL-X betas，做成 Chupa 可用的 subject pkl。
在 chupa 環境執行（或任何有 numpy 的環境）。
"""
import pickle
import numpy as np
import torch
from pathlib import Path

# 轉換結果
TRANSFER_PKL = Path.home() / "smplx_transfer_repo/output/user_shape.pkl"
# 模板：沿用 emily A-pose 的 pkl（取其 A-pose 姿態、scale、translation）
TEMPLATE_PKL = Path("data/renderpeople/smplx/rp_emily_apose_001.pkl")
# 輸出
OUT_PKL = Path("data/renderpeople/smplx/rp_user_apose_001.pkl")

def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)

# 讀轉換結果，取前 10 維 betas
with open(TRANSFER_PKL, "rb") as f:
    transferred = pickle.load(f)
user_betas = to_np(transferred["betas"]).reshape(-1)[:10]
print(f"使用者 SMPL-X betas (前10維): {user_betas.round(3)}")

# 讀模板，只換 betas（保留 A-pose 姿態與 scale/translation）
with open(TEMPLATE_PKL, "rb") as f:
    template = pickle.load(f, encoding="latin1")

data = dict(template)
orig = np.asarray(template["betas"])
data["betas"] = user_betas.astype(orig.dtype).reshape(orig.shape)

with open(OUT_PKL, "wb") as f:
    pickle.dump(data, f)

print(f"已建立: {OUT_PKL}")
print("\n下一步：用這個 subject 生成著衣網格")
print(f"  python scripts/chupa.py configs/text.yaml \\")
print(f"    dataset.subject=rp_user_apose_001 \\")
print(f"    'chupa.prompt=a woman wearing a long dress' \\")
print(f"    chupa.seed=42 chupa.text_steps=100 chupa.text_cfg_scale=7.5 \\")
print(f"    chupa.text_image_cfg_scale=1.5 \\")
print(f"    chupa.n_resample_from_text=0 chupa.n_resample=0")
