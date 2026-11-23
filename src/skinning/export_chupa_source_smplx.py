from pathlib import Path
import pickle
import numpy as np
import torch
import smplx
import trimesh


def to_tensor(data, key, default_shape):
    if key not in data:
        return torch.zeros(default_shape, dtype=torch.float32)

    arr = np.asarray(data[key]).astype(np.float32)

    if arr.ndim == 1:
        arr = arr[None]

    return torch.tensor(arr, dtype=torch.float32)


def main():
    smplx_model_dir = Path("src/smpl_related/models")
    smplx_param_path = Path("data/renderpeople/smplx/rp_ben_posed_001.pkl")
    out_path = Path("debug_source_smplx_rp_ben_posed_001.obj")

    if not smplx_param_path.exists():
        raise FileNotFoundError(smplx_param_path)

    print("[INFO] loading:", smplx_param_path)

    with open(smplx_param_path, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    print("[INFO] param keys:", params.keys())

    global_orient = to_tensor(params, "global_orient", (1, 3))
    body_pose = to_tensor(params, "body_pose", (1, 63))
    betas = to_tensor(params, "betas", (1, 10))
    transl = to_tensor(params, "transl", (1, 3))

    body_pose = body_pose[:, :63]
    betas = betas[:, :10]

    print("[INFO] global_orient:", global_orient.shape, global_orient.numpy())
    print("[INFO] body_pose:", body_pose.shape)
    print("[INFO] betas:", betas.shape)
    print("[INFO] transl:", transl.shape, transl.numpy())

    model = smplx.create(
        model_path=str(smplx_model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
        batch_size=1,
    )

    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
        )

    verts = output.vertices[0].detach().cpu().numpy()
    faces = model.faces

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(out_path)

    print("[DONE] saved:", out_path)
    print("[INFO] verts:", verts.shape)
    print("[INFO] faces:", faces.shape)
    print("[INFO] bounds:", mesh.bounds)


if __name__ == "__main__":
    main()
