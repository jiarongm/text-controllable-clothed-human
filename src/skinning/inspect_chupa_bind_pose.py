import pickle, numpy as np

with open("data/renderpeople/smplx/rp_ben_posed_001.pkl", "rb") as f:
    bind_params = pickle.load(f)

bind_body_pose = np.array(bind_params["body_pose"]).reshape(1, -1)   # 應為 (1, 63)  = 21 joints x 3
bind_global_orient = np.array(bind_params["global_orient"]).reshape(1, -1)  # (1, 3)
bind_betas = np.array(bind_params["betas"]).reshape(1, -1)
print(bind_body_pose.shape, bind_global_orient.shape, bind_betas.shape)

import smplx, torch

model = smplx.create("src/smpl_related/models", model_type="smplx",
                      gender="neutral", use_pca=False, num_betas=10, batch_size=1)

out = model(
    global_orient=torch.tensor(bind_global_orient, dtype=torch.float32),
    body_pose=torch.tensor(bind_body_pose, dtype=torch.float32),
    betas=torch.tensor(bind_betas, dtype=torch.float32),
    return_verts=True,
)
verts_check = out.vertices.detach().numpy()[0]

import trimesh
source_mesh = trimesh.load("debug_source_smplx_rp_ben_posed_001.obj", process=False)
diff = np.linalg.norm(verts_check - np.asarray(source_mesh.vertices), axis=1)
print("mean diff:", diff.mean(), "max diff:", diff.max())
