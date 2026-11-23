from pathlib import Path
import argparse
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial import cKDTree


def load_romp_results(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    if "results" not in data.files:
        raise KeyError(f"{npz_path} does not contain key 'results'. Keys: {data.files}")

    return data["results"].item()


def smooth_transfer_weights(source_verts, chupa_verts, smpl_lbs_weights, k=16, sigma=0.04):
    """
    使用 KNN 平滑轉移 SMPL skinning weights。
    source_verts: (6890, 3)
    chupa_verts: (N, 3)
    smpl_lbs_weights: (6890, 24)
    """
    tree = cKDTree(source_verts)
    dist, idx = tree.query(chupa_verts, k=k)

    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    # 距離越近權重越大
    w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)

    # smpl_lbs_weights[idx] -> (N, k, 24)
    transferred = (smpl_lbs_weights[idx] * w[:, :, None]).sum(axis=1)

    transferred = transferred / np.maximum(transferred.sum(axis=1, keepdims=True), 1e-8)

    return transferred.astype(np.float32), dist, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_npz", required=True, help="ROMP source npz，建議使用 result.npz")
    parser.add_argument("--chupa_mesh", required=True, help="Chupa mesh_refine.obj")
    parser.add_argument("--out_dir", default="animation_data_smpl")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])

    # 新增：KNN 平滑參數
    parser.add_argument("--k", type=int, default=16, help="Number of nearest SMPL vertices for smooth weight transfer")
    parser.add_argument("--sigma", type=float, default=0.04, help="Gaussian distance sigma for KNN weight smoothing")

    args = parser.parse_args()

    smpl_model_dir = Path("src/smpl_related/models")
    source_npz = Path(args.source_npz).expanduser()
    chupa_mesh_path = Path(args.chupa_mesh)
    out_dir = Path(args.out_dir)

    if not source_npz.exists():
        raise FileNotFoundError(source_npz)
    if not chupa_mesh_path.exists():
        raise FileNotFoundError(chupa_mesh_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading ROMP source npz:", source_npz)
    romp = load_romp_results(source_npz)

    if "verts" not in romp:
        raise KeyError(f"ROMP result does not contain 'verts'. Keys: {romp.keys()}")

    source_smpl_verts = romp["verts"]

    if source_smpl_verts.ndim != 3:
        raise ValueError(f"Unexpected ROMP verts shape: {source_smpl_verts.shape}")

    source_smpl_verts = source_smpl_verts[0].astype(np.float32)

    print("[INFO] source SMPL verts from ROMP:", source_smpl_verts.shape)

    if source_smpl_verts.shape[0] != 6890:
        raise ValueError(f"Expected SMPL verts = 6890, but got {source_smpl_verts.shape[0]}")

    print("[INFO] Loading Chupa mesh:", chupa_mesh_path)
    chupa_mesh = trimesh.load(chupa_mesh_path, force="mesh", process=False)
    chupa_vertices = np.asarray(chupa_mesh.vertices, dtype=np.float32)
    chupa_faces = np.asarray(chupa_mesh.faces, dtype=np.int64)

    print("[INFO] Chupa verts:", chupa_vertices.shape)
    print("[INFO] Chupa faces:", chupa_faces.shape)

    print("[INFO] Creating SMPL model...")
    smpl_model = smplx.create(
        model_path=str(smpl_model_dir),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        batch_size=1,
    )

    smpl_lbs_weights = smpl_model.lbs_weights.detach().cpu().numpy().astype(np.float32)

    print("[INFO] SMPL lbs_weights:", smpl_lbs_weights.shape)

    if smpl_lbs_weights.shape != (6890, 24):
        raise ValueError(f"Expected SMPL lbs_weights shape (6890, 24), got {smpl_lbs_weights.shape}")

    print("[INFO] Smooth KNN weight transfer")
    print("[INFO] k:", args.k)
    print("[INFO] sigma:", args.sigma)

    chupa_smpl_weights, dist, nearest = smooth_transfer_weights(
        source_verts=source_smpl_verts,
        chupa_verts=chupa_vertices,
        smpl_lbs_weights=smpl_lbs_weights,
        k=args.k,
        sigma=args.sigma,
    )

    print("[INFO] nearest distance:")
    print("  min :", float(dist.min()))
    print("  mean:", float(dist.mean()))
    print("  max :", float(dist.max()))

    print("[INFO] Chupa SMPL weights:", chupa_smpl_weights.shape)
    print("[INFO] weight sum min/max:",
          float(chupa_smpl_weights.sum(axis=1).min()),
          float(chupa_smpl_weights.sum(axis=1).max()))

    np.save(out_dir / "chupa_smpl_skinning_weights.npy", chupa_smpl_weights)
    np.save(out_dir / "chupa_template_vertices.npy", chupa_vertices)
    np.save(out_dir / "chupa_faces.npy", chupa_faces)
    np.save(out_dir / "chupa_nearest_smpl_vertex.npy", nearest)
    np.save(out_dir / "chupa_nearest_smpl_distance.npy", dist)

    print("[DONE] saved:")
    print(" ", out_dir / "chupa_smpl_skinning_weights.npy")
    print(" ", out_dir / "chupa_template_vertices.npy")
    print(" ", out_dir / "chupa_faces.npy")
    print(" ", out_dir / "chupa_nearest_smpl_vertex.npy")
    print(" ", out_dir / "chupa_nearest_smpl_distance.npy")


if __name__ == "__main__":
    main()