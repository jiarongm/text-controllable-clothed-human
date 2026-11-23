r"""
gradio_realtime_romp_chupa.py

在 romp 環境執行。

即時：Camera → ROMP → SMPL-X 蒙皮驅動 Chupa mesh（帶貼圖）
換裝：使用者輸入文字 prompt → 生成幾何 + UV 貼圖 → 重新蒙皮 → 替換即時串流資產

重要修正紀錄：
  1. body_pose 66 維要丟棄「前 3 個值」(slot 0)，取後 63 個
     （依據 src/smpl_related/utils.py 官方邏輯）
  2. 必須套用 body_scale 與 global_body_translation
  3. ROMP 相機空間與 SMPL-X 世界座標系差一個繞 X 軸 180 度旋轉
  4. Chupa 文字生成必須加 chupa.n_resample_from_text=0
     否則 resample 階段會用裸體先驗把文字生成的服裝結構洗掉

────────────────────────────────────────────────────────────────
加入貼圖後的兩個關鍵改變
────────────────────────────────────────────────────────────────
  * 貼圖流程用 xatlas 重新展開 UV，頂點數與順序都跟 mesh_refine.obj 不同。
    所以換裝時 v_bind_world / weights / faces / uv 必須整組一起換，
    而且權重要用「重新索引後」的頂點重算，不能沿用舊的。

  * 每影格匯出的 glb 現在帶貼圖。glTF 原生支援材質，只要把 UV 與貼圖掛到
    trimesh 物件上就會embed進去。代價是每影格要重新編碼一次 PNG，所以
    即時串流用的貼圖會先降到 LIVE_TEX_RES，避免拖慢串流。

────────────────────────────────────────────────────────────────
現場 demo 的現實
────────────────────────────────────────────────────────────────
  完整生成（幾何 + 貼圖）約 25-30 分鐘，快速模式約 12-15 分鐘。
  兩者都不適合讓使用者站在鏡子前面等。

  實際可行的做法是預生成服裝庫（比照你現有的 shape_assets）：
  先批次跑好一批常見服裝，使用者選擇時瞬間切換，只有自訂 prompt 才走
  完整生成。UI 上這兩條路徑分開呈現，讓使用者知道自己在等什麼。
"""

import os
import glob
import time
import pickle
import threading
import subprocess

import numpy as np
import torch
import smplx
import trimesh
import gradio as gr
import romp
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R_scipy
from smplx.lbs import (
    batch_rodrigues,
    vertices2joints,
    blend_shapes,
    batch_rigid_transform,
)

import json
import glob as _glob

import outfit_pipeline as op


SHAPE_ASSETS_DIR = "shape_assets"


# ============================================================
# 常數設定
# ============================================================

NUM_JOINTS = 55
NUM_BODY_JOINTS = 21

# ROMP 相機空間 -> SMPL-X 世界座標系校正（繞 X 軸 180 度）
R_CW = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

CHUPA_PYTHON = "/home/jiaro/miniconda3/envs/chupa/bin/python"
CHUPA_PROJECT_DIR = "/home/jiaro/projects/chupa"
CHUPA_SUBJECT = "rp_emily_apose_001"

SOURCE_SMPLX_OBJ_NAME = "debug_source_smplx_rp_emily_apose_001.obj"
ANIMATION_DATA_DIR = "animation_data_emily_apose"

OUTPUT_GLB_PATH = "/tmp/live_frame.glb"

# 即時串流用的貼圖解析度。每影格都要重新編碼 PNG 塞進 glb，
# 用 1024 會明顯拖慢串流；512 在鏡子距離上看不出差別。
LIVE_TEX_RES = 512

# trimesh 匯出 glTF 時是否已經幫你翻轉 V 軸。
# OBJ 的 v 原點在下、glTF 在上，兩者相反。如果模型出現「顏色種類正確
# 但位置全部錯亂」的拼貼亂碼，就是這個值設反了，改成 True 再試。
UV_FLIP_V = False

asset_lock = threading.Lock()


# ============================================================
# 蒙皮引擎核心函式
# ============================================================

def build_smplx_model(model_dir, gender="neutral", num_betas=10):
    model = smplx.create(model_dir, model_type="smplx", gender=gender,
                         use_pca=True, num_pca_comps=6,
                         num_betas=num_betas, batch_size=1)
    model.eval()
    return model


def make_full_pose(global_orient_3, body_pose_63):
    full_pose = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
    full_pose[0] = global_orient_3.reshape(3)
    full_pose[1:1 + NUM_BODY_JOINTS] = body_pose_63.reshape(NUM_BODY_JOINTS, 3)
    return full_pose


def compute_joint_transforms(model, betas, full_pose_55x3):
    """計算每個關節在指定姿勢下的變換矩陣，回傳 (55, 4, 4)。"""
    betas_t = torch.tensor(betas.reshape(1, -1), dtype=torch.float32)
    full_pose_t = torch.tensor(full_pose_55x3.reshape(1, NUM_JOINTS, 3), dtype=torch.float32)

    v_shaped = model.v_template.unsqueeze(0) + blend_shapes(
        betas_t, model.shapedirs[:, :, : betas_t.shape[-1]])
    J_rest = vertices2joints(model.J_regressor, v_shaped)

    rot_mats = batch_rodrigues(full_pose_t.view(-1, 3)).view(1, NUM_JOINTS, 3, 3)
    _, A = batch_rigid_transform(rot_mats, J_rest, model.parents)
    return A.squeeze(0).detach().numpy()


def reskin_vertices(v_bind, weights_arr, G_bind_arr, G_new_arr):
    """v_new[i] = sum_j weight[i,j] * (G_new[j] @ inv(G_bind[j])) @ v_bind[i]"""
    V = v_bind.shape[0]
    T = np.matmul(G_new_arr, np.linalg.inv(G_bind_arr))
    blended = np.einsum("vj,jab->vab", weights_arr, T)
    v_bind_homo = np.concatenate([v_bind, np.ones((V, 1), dtype=np.float32)], axis=1)
    return np.einsum("vab,vb->va", blended, v_bind_homo)[:, :3]


def world_to_raw(v_world, scale, translation):
    return (v_world - translation.reshape(1, 3)) / scale


def raw_to_world(v_raw, scale, translation):
    return v_raw * scale + translation.reshape(1, 3)


def load_bind_params(pkl_path):
    with open(pkl_path, "rb") as f:
        params = pickle.load(f, encoding="latin1")

    global_orient = np.array(params["global_orient"], dtype=np.float32).reshape(3)
    betas = np.array(params["betas"], dtype=np.float32).reshape(-1)
    bp66 = np.array(params["body_pose"], dtype=np.float32).reshape(22, 3)
    body_pose_63 = bp66[1:22].reshape(63)   # 丟棄 slot 0
    body_scale = float(np.asarray(params["body_scale"]).reshape(-1)[0])
    translation = np.asarray(params["global_body_translation"],
                             dtype=np.float32).reshape(3)

    return {"betas": betas,
            "full_pose": make_full_pose(global_orient, body_pose_63),
            "body_scale": body_scale,
            "translation": translation}


def romp_smpl_to_smplx_pose(romp_theta_72):
    """ROMP (72,) → SMPL-X 的 (global_orient, body_pose_63)，含座標系校正。

    body_pose 是相對父關節的局部旋轉，不受世界座標系影響，不校正。
    """
    romp_theta_72 = romp_theta_72.reshape(24, 3)
    romp_go_mat = R_scipy.from_rotvec(romp_theta_72[0]).as_matrix()
    global_orient = R_scipy.from_matrix(R_CW @ romp_go_mat).as_rotvec().astype(np.float32)
    body_pose_63 = romp_theta_72[1:22].reshape(63)   # 捨棄 SMPL joint 22,23 (hands)
    return global_orient, body_pose_63


def drive_chupa_with_romp_frame(model, bind_dict, weights_arr, v_bind_raw_arr,
                                 G_bind_arr, romp_theta_72):
    global_orient, body_pose_63 = romp_smpl_to_smplx_pose(romp_theta_72)
    new_full_pose = make_full_pose(global_orient, body_pose_63)
    G_new = compute_joint_transforms(model, bind_dict["betas"], new_full_pose)
    v_new_raw = reskin_vertices(v_bind_raw_arr, weights_arr, G_bind_arr, G_new)
    return raw_to_world(v_new_raw, bind_dict["body_scale"], bind_dict["translation"])


# ============================================================
# 權重轉移
# ============================================================

def transfer_weights(target_verts, source_smplx_obj=None):
    """M1（最近頂點）權重轉移。

    選 M1 不是因為指標最好，而是它的弱點在這個系統不會被觸發：
    綁定資產都是無自我接觸的 A-pose，而 M1 唯一的弱點就是自我接觸誤配。
    實測 M2（k-NN 高斯）在乾淨 A-pose 上驅動後只差 0.63mm（身高的 0.05%），
    視覺無法區分，卻要 4.6 倍時間（163ms vs 35ms）。

    如果之後改用有自我接觸的 bind pose 要換成 M2。注意 body-part 約束法
    若用單一最近點決定部位會完全退化成 M1（權重逐位元相同），
    必須用 k=12 多數決決定部位才會真的生效。
    """
    obj = source_smplx_obj or os.path.join(CHUPA_PROJECT_DIR, SOURCE_SMPLX_OBJ_NAME)
    source_mesh = trimesh.load(obj, force="mesh", process=False)
    source_verts = np.asarray(source_mesh.vertices, dtype=np.float32)

    lbs_weights = model.lbs_weights.detach().cpu().numpy().astype(np.float32)
    if lbs_weights.shape[0] != source_verts.shape[0]:
        raise ValueError(f"lbs_weights 頂點數 {lbs_weights.shape[0]} "
                         f"!= source verts {source_verts.shape[0]}")

    dist, nearest_idx = cKDTree(source_verts).query(target_verts, k=1)
    print(f"[INFO] nearest distance: mean={dist.mean():.4f} max={dist.max():.4f}")

    w = lbs_weights[nearest_idx]
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = (w / s).astype(np.float32)

    # 健康度檢查，避免把壞掉的權重換上去導致即時串流整個炸掉
    if np.isnan(w).any():
        raise ValueError("新權重含有 NaN，中止換裝")
    if not (np.abs(w.sum(axis=1) - 1.0) < 1e-3).all():
        raise ValueError("新權重每列總和不等於 1，中止換裝")
    return w


# ============================================================
# 貼圖
# ============================================================

def load_live_texture(path):
    """載入貼圖並降到 LIVE_TEX_RES。

    每影格匯出 glb 都要重新編碼一次 PNG，全解析度會明顯拖慢串流。
    先降好放在記憶體裡，串流時直接重用同一個 PIL Image 物件。
    """
    img = Image.open(path).convert("RGB")
    if max(img.size) > LIVE_TEX_RES:
        img = img.resize((LIVE_TEX_RES, LIVE_TEX_RES), Image.LANCZOS)
    return img


def make_visual(uv_arr, tex_img):
    """把 UV 與貼圖包成 trimesh 的材質物件。沒有 UV 就回傳 None（顯示灰模）。"""
    if uv_arr is None or tex_img is None:
        return None
    from trimesh.visual import TextureVisuals
    uv_use = uv_arr.copy()
    if UV_FLIP_V:
        uv_use[:, 1] = 1.0 - uv_use[:, 1]
    return TextureVisuals(uv=uv_use, image=tex_img)


# ============================================================
# 身形資產（沿用原有邏輯）
# ============================================================

def _load_all_shape_metas():
    metas = []
    for meta_path in _glob.glob(os.path.join(SHAPE_ASSETS_DIR, "*", "meta.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            m["_dir"] = os.path.dirname(meta_path)
            metas.append(m)
        except Exception as e:
            print(f"[WARN] 讀取 {meta_path} 失敗: {e}")
    return metas


def _find_nearest_shape(height_cm, weight_kg, metas):
    best, best_d = None, float("inf")
    for m in metas:
        dh = (height_cm - m["height_cm"]) / 10.0
        dw = (weight_kg - m["weight_kg"]) / 10.0
        d = dh * dh + dw * dw
        if d < best_d:
            best, best_d = m, d
    return best, best_d


def apply_nearest_shape(height_cm, weight_kg):
    global v_bind_world, weights, faces, v_bind_raw, G_bind, bind, uv, live_tex

    if not height_cm or not weight_kg:
        return "請輸入身高與體重"
    metas = _load_all_shape_metas()
    if not metas:
        return f"找不到任何預生成體型（請確認 {SHAPE_ASSETS_DIR}/ 存在）"

    best, _ = _find_nearest_shape(float(height_cm), float(weight_kg), metas)
    try:
        d = best["_dir"]
        new_v = np.load(f"{d}/template_vertices.npy").astype(np.float32)
        new_w = np.load(f"{d}/weights.npy").astype(np.float32)
        new_f = np.load(f"{d}/faces.npy").astype(np.int64)
        new_G = np.load(f"{d}/G_bind.npy")
        new_bind = {"betas": np.array(best["betas"], dtype=np.float32),
                    "body_scale": float(best["body_scale"]),
                    "translation": np.array(best["translation"], dtype=np.float32)}

        # 身形資產沒有貼圖（是換體型不是換衣服），所以清掉材質顯示灰模
        with asset_lock:
            v_bind_world, weights, faces, G_bind, bind = new_v, new_w, new_f, new_G, new_bind
            v_bind_raw = world_to_raw(new_v, new_bind["body_scale"], new_bind["translation"])
            uv, live_tex = None, None

        return (f"已套用最接近體型：{best['height_cm']}cm / {best['weight_kg']}kg "
                f"（您輸入 {height_cm}cm / {weight_kg}kg，服裝：{best.get('prompt', '')}）"
                f"\n注意：身形資產不含貼圖，會顯示灰模。要上色請重新生成服裝。")
    except Exception as e:
        return f"載入失敗: {e}"


# ============================================================
# 換裝
# ============================================================

def _swap_outfit(new_v, new_f, new_uv, new_w, tex_path):
    """原子性替換所有資產，避免即時串流讀到更新到一半的狀態。"""
    global v_bind_world, weights, faces, v_bind_raw, uv, live_tex
    tex_img = load_live_texture(tex_path) if tex_path else None
    new_raw = world_to_raw(new_v, bind["body_scale"], bind["translation"])
    with asset_lock:
        v_bind_world, faces, uv, weights = new_v, new_f, new_uv, new_w
        v_bind_raw = new_raw
        live_tex = tex_img


def regenerate_outfit(prompt, fast_mode, save_name, progress=gr.Progress()):
    """完整換裝：生成幾何 + 貼圖 + 重算權重。"""
    if not prompt or not prompt.strip():
        return "請輸入服裝描述文字"

    try:
        outfit = op.build_outfit(
            prompt=prompt.strip(),
            subject=CHUPA_SUBJECT,
            smplx_obj_name=SOURCE_SMPLX_OBJ_NAME,
            fast=bool(fast_mode),
            progress=progress,
        )

        progress(0.96, desc="計算蒙皮權重...")
        # 注意：要用 xatlas 重新索引後的頂點算權重，不能沿用舊的
        new_w = transfer_weights(outfit["verts"])

        _swap_outfit(outfit["verts"], outfit["faces"], outfit["uv"],
                     new_w, outfit["texture"])

        msg = (f"換裝完成：{prompt}\n"
               f"頂點數 {len(outfit['verts'])}，"
               f"模式：{'快速' if fast_mode else '完整'}")

        if save_name and save_name.strip():
            path = op.save_outfit_asset(outfit, new_w, save_name.strip())
            msg += f"\n已存成資產：{path}（下次可直接載入）"

        progress(1.0, desc="完成")
        return msg

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"生成失敗: {e}"


def refresh_outfit_list():
    items = op.list_outfit_assets()
    return gr.update(choices=[i[0] for i in items],
                     value=items[0][0] if items else None)


def apply_saved_outfit(choice):
    """載入預生成服裝，瞬間完成。"""
    if not choice:
        return "請先選擇一套服裝"
    items = dict(op.list_outfit_assets())
    if choice not in items:
        return f"找不到 {choice}，請按重新整理"
    try:
        v, f, u, w, tex = op.load_outfit_asset(items[choice])
        _swap_outfit(v, f, u, w, tex)
        return f"已套用：{choice}（頂點數 {len(v)}）"
    except Exception as e:
        return f"載入失敗: {e}"


# ============================================================
# 初始化
# ============================================================

print("[INFO] 初始化蒙皮引擎...")

smplx_model_dir = os.path.join(CHUPA_PROJECT_DIR, "src/smpl_related/models")
bind_pkl_path = os.path.join(CHUPA_PROJECT_DIR,
                             f"data/renderpeople/smplx/{CHUPA_SUBJECT}.pkl")

bind = load_bind_params(bind_pkl_path)
model = build_smplx_model(smplx_model_dir)
G_bind = compute_joint_transforms(model, bind["betas"], bind["full_pose"])

weights = np.load(os.path.join(CHUPA_PROJECT_DIR, ANIMATION_DATA_DIR,
                               "chupa_skinning_weights.npy")).astype(np.float32)
v_bind_world = np.load(os.path.join(CHUPA_PROJECT_DIR, ANIMATION_DATA_DIR,
                                    "chupa_template_vertices.npy")).astype(np.float32)
faces = np.load(os.path.join(CHUPA_PROJECT_DIR, ANIMATION_DATA_DIR,
                             "chupa_faces.npy")).astype(np.int64)
v_bind_raw = world_to_raw(v_bind_world, bind["body_scale"], bind["translation"])

# 初始資產沒有貼圖（是舊流程產生的），先顯示灰模，換裝後才有材質
uv = None
live_tex = None

print(f"[INFO] subject: {CHUPA_SUBJECT}")
print(f"[INFO] 頂點數: {v_bind_world.shape[0]}, 面數: {faces.shape[0]}")

print("[INFO] 初始化 ROMP...")
settings = romp.main.default_settings
settings.calc_smpl = True
romp_model = romp.ROMP(settings)

print("[INFO] 初始化完成")


# ============================================================
# 即時串流處理
# ============================================================

def process_frame(frame):
    if frame is None:
        return None

    outputs = romp_model(frame)
    if outputs is None or "smpl_thetas" not in outputs or len(outputs["smpl_thetas"]) == 0:
        return None

    romp_theta_72 = outputs["smpl_thetas"][0]

    with asset_lock:
        v_new_world = drive_chupa_with_romp_frame(
            model, bind, weights, v_bind_raw, G_bind, romp_theta_72)
        faces_snapshot = faces
        visual = make_visual(uv, live_tex)

    mesh = trimesh.Trimesh(vertices=v_new_world, faces=faces_snapshot,
                           visual=visual, process=False)
    mesh.export(OUTPUT_GLB_PATH)
    return OUTPUT_GLB_PATH


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks() as demo:
    gr.Markdown("## 即時虛擬穿衣鏡 — Camera → ROMP → 帶材質的 Chupa Mesh")

    with gr.Row():
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="攝影機輸入")
        mesh_output = gr.Model3D(label="即時驅動的著衣人體")

    webcam_input.stream(fn=process_frame, inputs=webcam_input, outputs=mesh_output)

    status_output = gr.Textbox(label="狀態", interactive=False, lines=3)

    with gr.Tab("快速換裝（預生成）"):
        gr.Markdown("從已經跑好的服裝庫直接載入，**瞬間完成**。")
        with gr.Row():
            outfit_dropdown = gr.Dropdown(
                choices=[i[0] for i in op.list_outfit_assets()],
                label="服裝庫")
            refresh_btn = gr.Button("重新整理", scale=0)
            apply_btn = gr.Button("套用", variant="primary", scale=0)

        refresh_btn.click(fn=refresh_outfit_list, outputs=outfit_dropdown)
        apply_btn.click(fn=apply_saved_outfit, inputs=outfit_dropdown,
                        outputs=status_output)

    with gr.Tab("自訂生成"):
        gr.Markdown(
            "輸入服裝描述會重新生成幾何與貼圖。\n\n"
            "**快速模式約 12-15 分鐘，完整模式約 25-30 分鐘。**\n"
            "顏色請用英文顏色詞加服裝名詞（例如 `a red jacket and black pants`），"
            "這樣才能正確分辨上衣與下著的顏色。")

        prompt_input = gr.Textbox(
            label="服裝描述",
            placeholder="a woman wearing a navy coat and khaki trousers")
        with gr.Row():
            fast_checkbox = gr.Checkbox(value=True, label="快速模式（品質略降）")
            save_name_input = gr.Textbox(
                label="存入服裝庫的名稱（留空則不存）",
                placeholder="navy_coat")
        regen_button = gr.Button("生成新服裝", variant="primary")

        gr.Examples(
            examples=[
                "a woman wearing a red dress",
                "a person wearing a white t-shirt and blue jeans",
                "a woman wearing a navy coat and khaki trousers",
                "a person wearing a green hoodie and black pants",
            ],
            inputs=prompt_input)

        regen_button.click(fn=regenerate_outfit,
                           inputs=[prompt_input, fast_checkbox, save_name_input],
                           outputs=status_output)

    with gr.Tab("體型"):
        gr.Markdown("套用最接近的預生成體型（**注意：體型資產不含貼圖，會顯示灰模**）。")
        with gr.Row():
            height_input = gr.Number(label="身高 (cm)", value=170)
            weight_input = gr.Number(label="體重 (kg)", value=65)
            shape_btn = gr.Button("套用最接近體型", scale=0)
        shape_btn.click(fn=apply_nearest_shape,
                        inputs=[height_input, weight_input],
                        outputs=status_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)