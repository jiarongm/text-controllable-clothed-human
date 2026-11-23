from pathlib import Path
import argparse
import numpy as np
import torch
import smplx
import trimesh


def load_romp_results(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "results" not in data.files:
        raise KeyError(f"{npz_path} does not contain key 'results'. Keys: {data.files}")
    return data["results"].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--zero_root", action="store_true")
    args = parser.parse_args()

    smpl_model_dir = Path("src/smpl_related/models")

    romp = load_romp_results(Path(args.npz).expanduser())

    if "smpl_thetas" in romp:
        pose = np.asarray(romp["smpl_thetas"][:1], dtype=np.float32)
    else:
        pose = np.concatenate(
            [
                np.asarray(romp["global_orient"][:1], dtype=np.float32),
                np.asarray(romp["body_pose"][:1], dtype=np.float32),
            ],
            axis=1,
        )

    if pose.shape[-1] != 72:
        raise ValueError(f"Expected pose dim 72, got {pose.shape}")

    if "smpl_betas" in romp:
        betas = np.asarray(romp["smpl_betas"][:1, :10], dtype=np.float32)
    else:
        betas = np.zeros((1, 10), dtype=np.float32)

    if args.zero_root:
        pose[:, 0:3] = 0.0
        print("[WARN] zero root orientation")

    print("[INFO] pose:", pose.shape)
    print("[INFO] betas:", betas.shape)
    print("[INFO] pose norm:", np.linalg.norm(pose))
    print("[INFO] root orient:", pose[:, 0:3])

    model = smplx.create(
        model_path=str(smpl_model_dir),
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
        batch_size=1,
    )

    global_orient = torch.tensor(pose[:, 0:3], dtype=torch.float32)
    body_pose = torch.tensor(pose[:, 3:72], dtype=torch.float32)
    betas = torch.tensor(betas, dtype=torch.float32)

    with torch.no_grad():
        out = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
        )

    verts = out.vertices[0].detach().cpu().numpy()
    faces = model.faces

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    print("[DONE] saved:", out_path)
    print("[INFO] verts:", verts.shape)
    print("[INFO] bounds:", mesh.bounds)


if __name__ == "__main__":
    main()
