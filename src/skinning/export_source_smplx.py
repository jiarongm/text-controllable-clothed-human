r"""
export_source_smplx.py —— 由 subject 的 pkl 匯出對齊的 SMPL-X 網格

`build_asset.py` 的 semantic 與 weights 兩個步驟都需要一份
`debug_source_smplx_<subject>.obj`——即該 subject 的人體先驗，**置於與 Chupa
輸出相同的世界座標**。既有的 `export_chupa_source_smplx_v2.py` 將 subject
寫死為 emily，故新建的 subject 無檔可用。本腳本為其參數化版本，邏輯逐行
沿用，僅將 subject 改為可指定。

五個易錯之處（沿用原檔的註記，因其皆為實際踩過的坑）
====================================================
1. `body_pose` 存為 66 維時須**丟棄前 3 個值**（slot 0），僅取 63 維。
2. 手部姿勢採 PCA，維度取前 6 維，須與 `src/smpl_related/utils.py` 一致。
3. 須套用 `body_scale` 與 `global_body_translation`，否則網格與 Chupa
   輸出不在同一座標系，語意標籤與權重轉移會整體錯位。
4. 建模時 `use_pca=True, num_pca_comps=6`，與 utils.py 一致。
5. yaw 不需另行處理——流程內部會自行抵銷。

以上任一項出錯，產物看起來仍是一具正常人體，**錯位要到驅動或合成時才會
顯現**，故不可省略核對。本腳本於結尾印出尺寸與質心供比對。

用法
====
  conda activate chupa
  cd ~/projects/chupa
  python export_source_smplx.py --subject rp_user_155_80_001

  # 指定輸出檔名（預設為 debug_source_smplx_<subject>.obj）
  python export_source_smplx.py --subject rp_user_155_80_001 --out <路徑>

環境：chupa
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SMPLX_DIR = ROOT / "src/smpl_related/models"
PKL_DIR = ROOT / "data/renderpeople/smplx"


def to_tensor(torch, data, key, default_shape):
    if key not in data:
        return torch.zeros(default_shape, dtype=torch.float32)
    arr = np.asarray(data[key]).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    return torch.tensor(arr, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description="由 subject pkl 匯出 SMPL-X 網格")
    ap.add_argument("--subject", required=True,
                    help="例如 rp_user_155_80_001")
    ap.add_argument("--out", default="",
                    help="預設 debug_source_smplx_<subject>.obj")
    ap.add_argument("--model-dir", default=str(SMPLX_DIR))
    args = ap.parse_args()

    import torch
    import smplx
    import trimesh

    pkl = PKL_DIR / f"{args.subject}.pkl"
    if not pkl.exists():
        sys.exit(f"[錯誤] 找不到 {pkl}")
    out_path = Path(args.out) if args.out else \
        ROOT / f"debug_source_smplx_{args.subject}.obj"
    if out_path.exists():
        print(f"[注意] {out_path} 已存在，將覆寫")

    with open(pkl, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    global_orient = to_tensor(torch, params, "global_orient", (1, 3))
    betas = to_tensor(torch, params, "betas", (1, 10))

    # 1. body_pose：66 維時丟棄 slot 0
    bp66 = to_tensor(torch, params, "body_pose", (1, 66))
    body_pose = bp66[:, 3:] if bp66.shape[1] == 66 else bp66

    # 2. 手部姿勢採 PCA 前 6 維
    lh = to_tensor(torch, params, "left_hand_pose", (1, 12))[:, :6]
    rh = to_tensor(torch, params, "right_hand_pose", (1, 12))[:, :6]

    jaw = to_tensor(torch, params, "jaw_pose", (1, 3))
    leye = to_tensor(torch, params, "leye_pose", (1, 3))
    reye = to_tensor(torch, params, "reye_pose", (1, 3))
    expr = to_tensor(torch, params, "expression", (1, 10))

    # 3. scale 與 translation
    body_scale = float(np.asarray(params["body_scale"]).reshape(-1)[0])
    transl = np.asarray(params["global_body_translation"],
                        dtype=np.float32).reshape(1, 3)

    print(f"[INFO] subject       : {args.subject}")
    print(f"[INFO] betas         : "
          f"{np.array2string(betas.numpy().reshape(-1), precision=3)}")
    print(f"[INFO] body_pose     : {tuple(body_pose.shape)}")
    print(f"[INFO] body_scale    : {body_scale}")
    print(f"[INFO] translation   : {transl.reshape(-1)}")

    # 4. 建模參數須與 utils.py 一致
    model = smplx.create(
        model_path=args.model_dir, model_type="smplx", gender="neutral",
        use_pca=True, num_pca_comps=6, num_betas=10,
        num_expression_coeffs=10, batch_size=1)

    with torch.no_grad():
        o = model(betas=betas, global_orient=global_orient,
                  body_pose=body_pose, left_hand_pose=lh,
                  right_hand_pose=rh, jaw_pose=jaw,
                  leye_pose=leye, reye_pose=reye, expression=expr)

    # 5. yaw 不另行處理
    verts = o.vertices[0].detach().cpu().numpy() * body_scale + transl

    trimesh.Trimesh(vertices=verts, faces=model.faces,
                    process=False).export(str(out_path))

    size = verts.max(0) - verts.min(0)
    print(f"\n[完成] {out_path}")
    print(f"[INFO] 尺寸 (max−min) : {size}")
    print(f"[INFO] 質心           : {verts.mean(0)}")
    print("\n核對：此處的尺寸與質心，應與同一 subject 之 Chupa 輸出網格相當"
          "（著衣網格略大於裸體）。\n若二者差距甚大，表示 scale 或 translation"
          "未正確套用，後續的語意標籤與\n權重轉移將整體錯位——而該錯位於本檔"
          "本身看不出來。")


if __name__ == "__main__":
    main()
