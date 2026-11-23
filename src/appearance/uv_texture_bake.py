r"""
uv_texture_bake.py

為 Chupa 輸出的無材質 clothed human mesh (mesh_refine.obj) 產生 UV 貼圖。

這個版本拆成四個 stage，因為幾何與生成的套件需求互斥：
  render   需要 nvdiffrast / xatlas / trimesh → 只能在 chupa 舊環境 (py3.8 + torch1.13)
  semantic 需要 nvdiffrast / trimesh / smplx  → chupa 環境（產生 UV 語意標籤圖）
  generate 需要新版 diffusers → 需要新 torch，只能在另一個環境
  export   只用 numpy + 純檔案寫入 → 哪個環境都能跑

中間透過 --work 指定的資料夾用檔案交換，兩個環境不用互相看到對方的套件。

────────────────────────────────────────────────────────────────
為什麼需要 semantic stage
────────────────────────────────────────────────────────────────
只給 SD 一句「紅外套黑褲子」而沒有任何空間約束時，每個視角都會自己猜一次紅色
該畫在哪，50 個互相矛盾的猜測平均起來就是紅色糊得到處都是、連臉上都有紅條紋。

semantic stage 借用 SMPL-X 的骨架語意解決這件事：chupa mesh 的每個頂點找到最近的
SMPL-X 頂點，該頂點的 dominant joint 決定它屬於哪個身體部位，再把部位對應到衣物
區域（軀幹+肩肘→上衣、髖膝踝→褲子、頭→皮膚頭髮、腕與手指→皮膚、腳掌→鞋）。
烘成 UV 標籤圖後，generate stage 用它平塗出一張「區域底色圖」餵給 SD 當初始影像，
SD 只能在既定邊界內加布料紋理與陰影，沒辦法把紅色畫到腿上。

已知限制：外套下擺會垂到髖關節以下，骨架邊界不等於衣服邊界，所以有 HEM_DROP
可調（見下方參數說明）。要真正精準的衣物邊界得跑 human parsing 模型（如 SCHP），
那需要額外下載權重，目前沒有內建。

────────────────────────────────────────────────────────────────
環境準備
────────────────────────────────────────────────────────────────

  # A. 幾何環境：沿用現有的 chupa（已經有 nvdiffrast/xatlas/trimesh）
  #    不需要 diffusers，如果之前裝壞了可以移除：
  #    pip uninstall -y diffusers        （選擇性）

  # B. 生成環境：新開一個
  conda create -n chupa_tex python=3.10 -y
  conda activate chupa_tex
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  pip install diffusers transformers accelerate opencv-python numpy
  # 裝完確認 GPU 抓得到：
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  # 如果 cuda.is_available() 是 False，先看 nvidia-smi 的 driver 版本，
  # 可能要換成別的 cuXXX build，把訊息回報出來再挑。

────────────────────────────────────────────────────────────────
跑法
────────────────────────────────────────────────────────────────

  # 1) 幾何：UV 展開 + 渲染所有視角（chupa 環境，需要 GPU）
  conda activate chupa
  python uv_texture_bake.py --stage render \
      --mesh results/.../mesh_refine.obj --work tex_work

  # 2) 語意標籤圖（chupa 環境，很快）
  python uv_texture_bake.py --stage semantic --work tex_work \
      --smplx-obj debug_source_smplx_rp_ben_posed_001_v2.obj \
      --prompt "a person wearing a red jacket and black pants"
  #    → 打開 tex_work/label_preview.png 確認上衣/褲子邊界，
  #      不對就調 HEM_DROP 重跑這個 stage（幾秒鐘的事）

  # 3) 先 --dry-run：不載入 SD，直接輸出平塗底色版本，
  #    等於「沒有布料細節但分區正確」的結果，用來確認邊界烘到 mesh 上對不對
  conda activate chupa_tex
  python uv_texture_bake.py --stage generate --work tex_work --dry-run
  #    → 用 Blender 開 uv_texture_output/chupa_textured.obj，
  #      應該看到乾淨的紅上衣 / 黑褲子 / 膚色頭手，沒有任何顏色越界

  # 4) 正式生成（chupa_tex 環境，需要 GPU）
  python uv_texture_bake.py --stage generate --work tex_work \
      --prompt "a person wearing a red jacket and black pants"

  # 5) 輸出帶材質的 obj（generate 會自動接，通常不用單獨跑）
  python uv_texture_bake.py --stage export --work tex_work --out uv_texture_output

────────────────────────────────────────────────────────────────
需要依實際狀況微調的地方
────────────────────────────────────────────────────────────────
  - HEM_DROP：外套下擺長度。紅色只到腰就斷掉 → 調大；蓋到大腿中段 → 調小。
    只影響 semantic stage，重跑很快，值得多試幾次調到滿意再跑 generate。
  - COLOR_WORDS / GARMENT_WORDS：區域底色是從 --prompt 自動解析出來的
    （"a red jacket and black pants" → 上衣紅、褲子黑）。prompt 用了詞彙表裡
    沒有的顏色詞就加進 COLOR_WORDS。解析不到會退回灰色並印出警告，不會壞掉。
  - SD_STRENGTH：SD 對平塗底色的改動幅度。太低像塑膠玩具沒布料感，
    太高紅色又會開始越界。0.75~0.85 通常是好範圍。
  - CAMERA_RADIUS_SCALE / FOVY_DEG：人被裁到畫面外或太小就調這兩個（影響 render stage）。
  - NORMAL_FLIP：如果生成結果的衣服皺褶凹凸感明顯反了，把對應軸的 1.0 改成 -1.0
    （影響 render stage，改完要重跑 render）。
  - ELEVATIONS / N_AZIMUTH：視角數，越多覆蓋率越好但 render 與 generate 都會變慢。
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# 參數設定（比照 compare_skinning_methods.py 的慣例）
# ============================================================

CHUPA_MESH = ("results/renderpeople/20260604-143444/rp_ben_posed_001/"
              "a_person_wearing_a_red_jacket_and_black_pants_0042/mesh_refine.obj")
PROMPT = "a person wearing a red jacket and black pants"
WORK_DIR = "tex_work"
OUT_DIR = "uv_texture_output"

# semantic stage 用（沿用 compare_skinning_methods.py 裡的同名路徑）
SMPLX_MODEL_DIR = "src/smpl_related/models"
SMPLX_SOURCE_OBJ = "debug_source_smplx_rp_ben_posed_001_v2.obj"

# 互動式使用（Gradio demo）時 25-30 分鐘的等待是不可行的。
# 設環境變數 UVBAKE_FAST=1 會砍到 12 個視角、較少推論步數，
# 貼圖生成縮到 3-5 分鐘。背面與遮擋區品質下降，正面觀感差異不大。
FAST = os.environ.get("UVBAKE_FAST", "") == "1"

TEX_RES = 1024            # UV texture 解析度
RENDER_RES = 768          # 每個視角的渲染解析度
N_AZIMUTH = 6 if FAST else 10            # 環繞水平視角數
# 額外仰角/俯角（弧度）。0.35/-0.35 約 20 度補髖部/下巴下方，
# 1.1/-1.1 約 63 度補頭頂/肩膀上緣、大腿內側/膝蓋後方。
# 不要太接近 ±90 度（look_at 用 world-up 當參考向量，接近極點會數值不穩定）。
ELEVATIONS = [0.0, 0.35] if FAST else [0.0, 0.35, -0.35, 1.1, -1.1]
SD_STEPS = 15 if FAST else 25

FOVY_DEG = 35.0
NEAR = 0.05
FAR = 5.0
CAMERA_RADIUS_SCALE = 1.3   # 相機到人體中心的距離倍率

NORMAL_FLIP = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# 匯出 OBJ 時是否翻轉 UV 的 V 軸。
#
# 實測兩個方向在 Blender 都不正確，所以這個開關目前解決不了問題（見 stage_export
# 的說明）。維持 False。要在 Blender 檢視結果請用同時輸出的 .glb，那條路徑
# 經過驗證是正確的。
OBJ_FLIP_V = False

CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_normalbae"
SD_MODEL = "runwayml/stable-diffusion-v1-5"
SD_INFER_RES = 512

# SD 對「已有內容」的改動幅度。0.75~0.85 通常是好範圍：
# 太低 (<0.5) 生成結果會被平塗底色綁死、看起來像塑膠玩具沒有布料感；
# 太高 (>0.95) SD 會開始無視底色的區域劃分，紅色又會亂跑。
SD_STRENGTH = 0.8


# ============================================================
# semantic stage 用的區域定義
# ============================================================

# SMPL-X 55 個關節 → 衣物區域。索引語意（SMPL-X 標準順序）：
#   0 pelvis, 1/2 hip, 3/6/9 spine, 4/5 knee, 7/8 ankle, 10/11 foot,
#   12 neck, 13/14 collar, 15 head, 16/17 shoulder, 18/19 elbow, 20/21 wrist,
#   22 jaw, 23/24 eye, 25-39 左手指, 40-54 右手指
REGION_OF_JOINT = {}
for _j in range(55):
    REGION_OF_JOINT[_j] = "upper"                       # 預設：軀幹歸上衣
for _j in [1, 2, 4, 5, 7, 8]:                            # 髖/膝/踝
    REGION_OF_JOINT[_j] = "lower"
for _j in [10, 11]:                                      # 腳掌
    REGION_OF_JOINT[_j] = "shoes"
for _j in [15, 22, 23, 24]:                              # 頭/下巴/眼
    REGION_OF_JOINT[_j] = "head"
for _j in [20, 21] + list(range(25, 55)):                # 腕 + 手指
    REGION_OF_JOINT[_j] = "hands"

REGION_IDS = {"upper": 1, "lower": 2, "head": 3, "hands": 4, "shoes": 5}

# 找不到顏色描述時的預設底色 (RGB)。
# 這些只是 SD 的初始影像，作用是把「哪裡是上衣、哪裡是褲子」的空間劃分
# 告訴 SD，實際色調 SD 還會依 prompt 調整，所以猜得不準也不致命。
DEFAULT_REGION_COLORS = {
    "upper": (120, 120, 125),
    "lower": (70, 70, 75),
    "head":  (200, 160, 130),    # 膚色（頭髮由 SD 依 prompt 自行畫上）
    "hands": (200, 160, 130),    # 膚色
    "shoes": (45, 42, 40),
}

# 英文顏色詞 → RGB。刻意取中等明度的版本：底色只是給 SD 的起點，
# 太飽和或太極端會讓 SD 難以在上面疊出合理的陰影與布料紋理。
COLOR_WORDS = {
    "black": (35, 35, 40), "white": (232, 230, 225), "grey": (128, 128, 130),
    "gray": (128, 128, 130), "silver": (180, 182, 185), "red": (170, 30, 30),
    "crimson": (150, 25, 40), "maroon": (100, 30, 35), "burgundy": (105, 30, 45),
    "pink": (225, 150, 170), "orange": (215, 115, 40), "yellow": (220, 195, 70),
    "gold": (190, 160, 60), "beige": (215, 200, 175), "cream": (235, 225, 200),
    "brown": (110, 80, 55), "tan": (185, 150, 110), "khaki": (170, 155, 110),
    "olive": (110, 110, 60), "green": (60, 130, 70), "emerald": (35, 130, 90),
    "teal": (40, 120, 125), "cyan": (90, 190, 200), "blue": (55, 85, 165),
    "navy": (35, 45, 85), "denim": (75, 100, 145), "indigo": (60, 55, 130),
    "purple": (110, 65, 150), "violet": (130, 90, 180), "lavender": (190, 175, 220),
}

# 每個區域會被哪些服裝名詞觸發。解析時從顏色詞往後找最近的服裝名詞，
# 判斷這個顏色是在講上衣、下著還是鞋。
#
# "full" 是連身服裝：洋裝、連身褲、長袍這類同時覆蓋上半身與下半身的衣物。
# 它們不屬於 upper 或 lower 任一者，而是要讓兩個區域取同一個顏色 ——
# 這是必要的特例，否則「a red dress」會解析不到任何區域顏色，
# 兩區都退回灰底，SD 失去分區引導，紅色會散得到處都是。
GARMENT_WORDS = {
    "upper": ["jacket", "coat", "shirt", "t-shirt", "tshirt", "tee", "sweater",
              "sweatshirt", "hoodie", "blouse", "top", "pullover", "cardigan",
              "jersey", "vest", "blazer", "parka", "windbreaker", "tunic"],
    "lower": ["pants", "trousers", "jeans", "shorts", "skirt", "leggings",
              "slacks", "chinos", "joggers", "sweatpants"],
    "shoes": ["shoes", "sneakers", "boots", "trainers", "loafers", "sandals",
              "heels", "footwear"],
    "full":  ["dress", "gown", "jumpsuit", "romper", "overall", "overalls",
              "robe", "kimono", "sari", "cheongsam", "qipao", "tunic-dress",
              "bodysuit", "onesie", "coverall", "frock"],
}

# 連身服裝要展開到哪些區域
FULL_BODY_REGIONS = ["upper", "lower"]


def parse_prompt_colors(prompt):
    """從服裝 prompt 推導各區域底色。

    做法是掃描 prompt 裡的顏色詞，往後找最近的服裝名詞來決定這個顏色屬於
    哪個區域。「a red jacket and black pants」→ red 接 jacket 歸 upper，
    black 接 pants 歸 lower。

    這只處理直白的顏色描述。像「千鳥格紋西裝」這種沒有單一顏色詞的描述會
    落回預設灰色 —— 那不是災難，因為底色只是 SD 的起點，SD 仍會依 prompt
    畫出花紋，只是分區的顏色引導會弱一些。
    """
    colors = dict(DEFAULT_REGION_COLORS)
    if not prompt:
        return colors, {}

    words = [w.strip(".,;:!?()[]\"'").lower() for w in prompt.split()]
    found = {}
    for i, w in enumerate(words):
        if w not in COLOR_WORDS:
            continue
        # 往後看幾個字，找最近的服裝名詞（"red jacket" / "red wool jacket" 都能接上）
        matched = False
        for j in range(i + 1, min(i + 5, len(words))):
            for region, nouns in GARMENT_WORDS.items():
                if words[j] not in nouns:
                    continue
                # 連身服裝展開到上下兩區；其餘只設自己那一區。
                # 已經被明確指定過的區域不覆寫 —— 「a red dress with a black belt」
                # 這種情況以先出現的為準。
                targets = FULL_BODY_REGIONS if region == "full" else [region]
                for t in targets:
                    if t not in found:
                        found[t] = w
                        colors[t] = COLOR_WORDS[w]
                matched = True
                break
            if matched:
                break
    return colors, found

# ── 外套下擺的判定 ────────────────────────────────────────────
# HEM_DROP —— 高度門檻。從腿部區域最高點往下推「身高的幾倍」，
# 這條線以上的腿部頂點改判成上衣。缺點是它本質上是一個水平面，
# 切出來的下擺偏直，不會完全跟著布料起伏（靠後面的標籤平滑稍微緩解）。
# 症狀：紅色只到腰就斷掉 → 調大；紅色蓋到大腿中段 → 調小。
HEM_DROP = 0.08

# ── 衣物厚度信號（預設關閉）─────────────────────────────────
# 原本的想法是：chupa 雕出了衣服的實際形狀，所以「chupa 表面到 SMPL-X 體表
# 的距離」應該就是衣物厚度，可以用來找出真正的衣服邊界。
#
# 實測發現這個假設在 chupa 的資料上不成立：chupa 的 SMPL-X 是對著「穿衣服的
# 人」擬合的，體表本來就貼在衣服表面附近，所以這個距離量到的是雕刻殘差，
# 不是衣物厚度。實測某件合身毛衣的分佈是 upper 0.0033 / lower 0.0043 /
# hands 0.0029 —— 外套沒有比褲子厚，皮膚也沒有比衣服薄，完全無法區分。
#
# 保留這段程式與診斷表，是因為它對「寬鬆衣物 + 貼合裸體的 SMPL-X」這類
# 資料仍可能有效（例如厚外套、裙子）。跑 semantic stage 時一定會印出厚度
# 分佈表，先看那張表確認各區域真的分得開，再決定要不要打開。
# 判斷標準：upper 的中位數要明顯大於 lower，hands 要明顯小於兩者。
USE_OFFSET_HEM = False           # 用厚度找下擺（取代 HEM_DROP）
USE_OFFSET_SKIN = False          # 用厚度找裸露皮膚
JACKET_OFFSET = 0.010            # 超過這個厚度才算被外套蓋住
SKIN_OFFSET = 0.004              # 低於這個厚度算裸露皮膚

# 標籤平滑：在網格上做鄰域多數決，讓邊界走自然路徑而不是鋸齒。
# 太大會把小區域（手指、鞋帶）吃掉，3~8 通常剛好。
LABEL_SMOOTH_ITERS = 5


# ============================================================
# Stage 1: render —— UV 展開 + 多視角渲染
# 只有這個 stage 需要 nvdiffrast / xatlas / trimesh
# ============================================================

def look_at(eye, target, up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """回傳 camera-to-world pose（OpenGL 慣例，相機看向 -Z）。
    假設人體是 Y-up（SMPL-X 常見慣例）。
    """
    z = eye - target
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def compute_camera_poses(mesh, n_azimuth, elevations):
    """回傳 [(pose, azimuth, elevation), ...]。
    方位角與仰角要一起回傳，因為 generate stage 需要靠它們決定生成順序
    （水平的正面/背面當錨點先生成，定下整體色調，其餘視角才補縫）。
    """
    center = mesh.bounding_box.centroid.astype(np.float32)
    radius = float(np.linalg.norm(mesh.bounding_box.extents)) * CAMERA_RADIUS_SCALE
    poses = []
    for elev in elevations:
        for i in range(n_azimuth):
            az = 2.0 * np.pi * i / n_azimuth
            eye = center + radius * np.array([
                np.cos(az) * np.cos(elev),
                np.sin(elev),
                np.sin(az) * np.cos(elev),
            ], dtype=np.float32)
            poses.append((look_at(eye, center), float(az), float(elev)))
    return poses


def make_projection(fovy_deg, aspect, near, far):
    """標準透視投影矩陣，y 項取負以符合 nvdiffrast 的影像列排列方向。"""
    fovy = np.deg2rad(fovy_deg)
    y = 1.0 / np.tan(fovy / 2.0)
    x = y / aspect
    return np.array([
        [x, 0, 0, 0],
        [0, -y, 0, 0],
        [0, 0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def pose_to_view_matrix(pose):
    R = pose[:3, :3]
    t = pose[:3, 3]
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R.T
    view[:3, 3] = -R.T @ t
    return view


def stage_render(mesh_path, work_dir):
    import trimesh
    import xatlas
    import torch
    import nvdiffrast.torch as dr

    work = Path(work_dir)
    views_dir = work / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 載入 mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    print("[INFO] UV 展開 (xatlas) ...")
    # chupa mesh 沒有原生 UV。xatlas 會依 UV 島切割重新索引頂點，
    # 之後所有 stage 都以這份重新索引後的網格為準。
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    verts = mesh.vertices[vmapping].astype(np.float32)
    normals = mesh.vertex_normals[vmapping].astype(np.float32)
    faces = indices.astype(np.int32)
    uv = uvs.astype(np.float32)
    print(f"       頂點數 {len(verts)}, 三角面數 {len(faces)}")

    np.savez_compressed(work / "geometry.npz", verts=verts, faces=faces, uv=uv)

    verts_t = torch.tensor(verts, dtype=torch.float32, device="cuda")
    faces_t = torch.tensor(faces, dtype=torch.int32, device="cuda")
    ctx = dr.RasterizeCudaContext()

    # UV atlas 有效面積 mask：把 UV 當 clip-space xy 光柵化，標出哪些 texel 真的
    # 屬於某個三角面。這是算「真實表面覆蓋率」的正確分母 —— 不然 xatlas 打包時
    # 島與島之間留的空隙也會被誤算成「沒覆蓋到」，讓數字看起來比實際嚴重很多。
    uv_clip = np.zeros((len(uv), 4), dtype=np.float32)
    uv_clip[:, 0] = uv[:, 0] * 2.0 - 1.0
    uv_clip[:, 1] = -(uv[:, 1] * 2.0 - 1.0)   # 對齊 texel 的 (1-v) 列排列慣例
    uv_clip[:, 3] = 1.0
    rast_atlas, _ = dr.rasterize(ctx, torch.tensor(uv_clip, device="cuda").unsqueeze(0),
                                 faces_t, resolution=[TEX_RES, TEX_RES])
    atlas_mask = (rast_atlas[0, ..., 3] > 0).detach().cpu().numpy()
    cv2.imwrite(str(work / "atlas_mask.png"), atlas_mask.astype(np.uint8) * 255)
    print(f"[INFO] UV atlas 實際佔用面積: {100.0 * atlas_mask.sum() / atlas_mask.size:.1f}% "
          f"(其餘是 xatlas 打包留的空隙，跟表面覆蓋率無關)")

    poses = compute_camera_poses(mesh, N_AZIMUTH, ELEVATIONS)
    proj = make_projection(FOVY_DEG, 1.0, NEAR, FAR)
    print(f"[INFO] 開始渲染 {len(poses)} 個視角 ...")

    verts_h = torch.cat(
        [verts_t, torch.ones(len(verts_t), 1, device="cuda", dtype=torch.float32)], dim=1)

    n_saved = 0
    view_info = []
    for view_i, (cam_pose, az, elev) in enumerate(poses):
        view = pose_to_view_matrix(cam_pose)
        mvp_t = torch.tensor(proj @ view, dtype=torch.float32, device="cuda")
        verts_clip = (verts_h @ mvp_t.T).unsqueeze(0)

        normal_cam = normals.dot(view[:3, :3].T)               # 世界法向 -> 相機空間
        facing = np.clip(normal_cam[:, 2], 0.0, 1.0)            # 面向相機程度 (0~1)

        uvw = np.concatenate([uv, facing[:, None]], axis=1).astype(np.float32)
        nrm = (normal_cam * NORMAL_FLIP * 0.5 + 0.5).astype(np.float32)

        rast, _ = dr.rasterize(ctx, verts_clip, faces_t, resolution=[RENDER_RES, RENDER_RES])
        uvw_img, _ = dr.interpolate(torch.tensor(uvw, device="cuda").unsqueeze(0), rast, faces_t)
        nrm_img, _ = dr.interpolate(torch.tensor(nrm, device="cuda").unsqueeze(0), rast, faces_t)
        uvw_img = uvw_img[0].detach().cpu().numpy()
        nrm_img = nrm_img[0].detach().cpu().numpy()
        mask = (rast[0, ..., 3] > 0).detach().cpu().numpy()

        if not mask.any():
            print(f"[view {view_i + 1}/{len(poses)}] 看不到 mesh，跳過")
            continue

        # 直接在這裡把 UV 座標換算成 texel 索引存成 uint16，
        # 這樣交換檔小很多，generate stage 也不必再知道 UV 的換算慣例。
        tx = np.clip((uvw_img[..., 0] * TEX_RES), 0, TEX_RES - 1).astype(np.uint16)
        ty = np.clip(((1.0 - uvw_img[..., 1]) * TEX_RES), 0, TEX_RES - 1).astype(np.uint16)
        facing_img = uvw_img[..., 2].astype(np.float16)

        np.savez_compressed(views_dir / f"view_{view_i:03d}.npz",
                            tx=tx, ty=ty, facing=facing_img, mask=mask)

        # normal map 背景先填中性白，避免誤導 ControlNet
        nrm_png = np.clip(nrm_img * 255.0, 0, 255).astype(np.uint8)
        nrm_png[~mask] = 255
        cv2.imwrite(str(views_dir / f"view_{view_i:03d}_normal.png"),
                    cv2.cvtColor(nrm_png, cv2.COLOR_RGB2BGR))
        view_info.append({"index": view_i, "azimuth": az, "elevation": elev})
        n_saved += 1

        if (view_i + 1) % 10 == 0 or view_i == len(poses) - 1:
            print(f"[view {view_i + 1}/{len(poses)}] 已存 {n_saved} 個視角")

    meta = {"tex_res": TEX_RES, "render_res": RENDER_RES,
            "n_views": len(poses), "n_saved": n_saved,
            "mesh_path": str(mesh_path), "sd_infer_res": SD_INFER_RES,
            "sd_strength": SD_STRENGTH, "n_azimuth": N_AZIMUTH,
            "controlnet_model": CONTROLNET_MODEL, "sd_model": SD_MODEL,
            "region_ids": REGION_IDS,
            "views": view_info}
    with open(work / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = sum(p.stat().st_size for p in work.rglob("*")) / 1e6
    print(f"\n[DONE] render stage 完成，交換檔在 {work}/ (約 {size_mb:.0f} MB)")
    print(f"       接著在同一個 chupa 環境跑 semantic stage 產生語意標籤圖：")
    print(f"       python uv_texture_bake.py --stage semantic --work {work} \\")
    print(f"           --smplx-obj <對齊到 chupa mesh 的 SMPL-X obj>")


# ============================================================
# Stage 2: semantic —— 產生 UV 語意標籤圖
# 需要 nvdiffrast / trimesh / smplx（chupa 環境）
# ============================================================

def stage_semantic(work_dir, smplx_obj, smplx_model_dir, prompt=None, no_smooth=False):
    import trimesh
    import torch
    import nvdiffrast.torch as dr
    from scipy.spatial import cKDTree

    work = Path(work_dir)
    g = np.load(work / "geometry.npz")
    verts, faces, uv = g["verts"], g["faces"], g["uv"]

    print(f"[INFO] 載入對齊的 SMPL-X mesh: {smplx_obj}")
    src = trimesh.load(smplx_obj, force="mesh", process=False)
    src_v = np.asarray(src.vertices, np.float32)

    # SMPL-X 的 LBS 權重：每個 SMPL-X 頂點的 dominant joint 決定它屬於哪個部位。
    # 這裡只需要權重矩陣，不做任何 forward kinematics。
    import smplx as smplx_lib
    model = smplx_lib.create(smplx_model_dir, model_type="smplx", gender="neutral",
                             use_pca=True, num_pca_comps=6, num_betas=10,
                             num_expression_coeffs=10, batch_size=1)
    lbs_w = model.lbs_weights.detach().cpu().numpy()
    if len(lbs_w) != len(src_v):
        raise ValueError(
            f"SMPL-X obj 有 {len(src_v)} 個頂點，但模型權重是 {len(lbs_w)} 個。\n"
            f"--smplx-obj 必須是未經 remesh 的 SMPL-X 拓樸（頂點順序要跟模型一致），\n"
            f"通常是你 pipeline 裡那個 debug_source_smplx_*.obj。")

    src_region = np.array([REGION_OF_JOINT[j] for j in lbs_w.argmax(axis=1)])

    # chupa 頂點 → 最近的 SMPL-X 頂點 → 繼承其區域標籤。
    #
    # 這裡不能只用全域最近點：手插口袋時手掌貼著外套，手部頂點的最近 SMPL-X 點
    # 可能落在軀幹上，於是手被判成上衣、渲染出來就是手上一塊紅斑。
    # 解法跟 compare_skinning_methods.py 的 M4 一樣：先用全域最近點決定大概部位，
    # 再限制只在同部位的來源頂點裡找最近點，避免跨部位誤配。
    print("[INFO] 建立 chupa → SMPL-X 對應（k=12 多數決定部位，避免自我接觸誤配）...")
    tree = cKDTree(src_v)

    # 部位判定一定要用 k 近鄰多數決，不能用單一最近點。
    # 用單一最近點的話，後面「限制在同部位裡找最近點」必然找回同一個頂點，
    # 整個約束會完全退化成純最近點法（結果逐位元相同），等於白做。
    # 在手插口袋那類自我接觸案例上，k=12 多數決會讓約 2400 個頂點的判定不同。
    _, knn = tree.query(verts, k=12)
    parts = np.unique(src_region)
    votes = np.zeros((len(verts), len(parts)), np.int32)
    for pi, part in enumerate(parts):
        votes[:, pi] = (src_region[knn] == part).sum(axis=1)
    region = parts[votes.argmax(axis=1)].astype("<U8")

    # 決定部位後，限制只在同部位的來源頂點裡找最近點
    idx0 = np.zeros(len(verts), np.int64)
    for part in parts:
        sel = np.where(region == part)[0]
        pool = np.where(src_region == part)[0]
        if len(sel) == 0 or len(pool) == 0:
            continue
        _, local = cKDTree(src_v[pool]).query(verts[sel], k=1)
        idx0[sel] = pool[local]
    region = src_region[idx0].astype("<U8")

    # 衣物厚度：chupa 表面到 SMPL-X 體表的距離。
    # 這張表永遠會印，因為它是判斷「厚度信號在這份資料上到底能不能用」的依據。
    # 能用的條件：upper 中位數明顯大於 lower，且 hands 明顯小於兩者。
    print("[INFO] 計算衣物厚度（chupa 表面 → SMPL-X 體表距離）...")
    _, dist_to_body, _ = trimesh.proximity.ProximityQuery(src).on_surface(verts)
    y = verts[:, 1]     # 假設 Y-up；座標系是 Z-up 的話改成 verts[:, 2]
    height = float(y.max() - y.min())
    offset = dist_to_body / max(height, 1e-8)      # 正規化成「佔身高的比例」

    med = {}
    print("\n       各區域衣物厚度分佈（佔身高比例）")
    print(f"       {'區域':<8}{'中位數':>10}{'75%':>10}{'90%':>10}")
    for name in REGION_IDS:
        m = region == name
        if m.any():
            q = np.percentile(offset[m], [50, 75, 90])
            med[name] = q[0]
            print(f"       {name:<8}{q[0]:>10.4f}{q[1]:>10.4f}{q[2]:>10.4f}")

    # 自動判斷這個信號在這份資料上有沒有鑑別力，免得預設值被誤用。
    usable = (med.get("upper", 0) > 1.3 * med.get("lower", 1e9)
              and med.get("hands", 1e9) < 0.7 * med.get("upper", 0))
    if usable:
        print("       → 厚度信號看起來有鑑別力，可考慮打開 USE_OFFSET_HEM/SKIN")
    else:
        print("       → 厚度信號在這份資料上無鑑別力（upper 沒有明顯厚於 lower，")
        print("         或 hands 沒有明顯薄於衣服）。這通常表示 SMPL-X 是對著")
        print("         穿衣服的人擬合的，量到的是雕刻殘差而非衣物厚度。")
        print("         維持 USE_OFFSET_* = False 是正確的。")
    print(f"       目前設定 USE_OFFSET_HEM={USE_OFFSET_HEM} "
          f"USE_OFFSET_SKIN={USE_OFFSET_SKIN}\n")

    # 外套下擺：骨架把大腿上緣判成 lower，但外套下擺實際蓋在上面。
    if (region == "lower").any():
        n_before = int((region == "lower").sum())
        if USE_OFFSET_HEM:
            # 用厚度判定：腿部區域裡厚度超過門檻的，是被外套蓋住的部分。
            # 邊界自然跟著布料起伏，不會是一條水平直線。
            # 仍加一個高度上限，避免褲腳堆疊處的厚度被誤判成外套。
            leg_top = float(y[region == "lower"].max())
            plausible = y > (leg_top - 0.25 * height)
            region[(region == "lower") & (offset > JACKET_OFFSET) & plausible] = "upper"
            print(f"[INFO] 下擺（厚度判定 JACKET_OFFSET={JACKET_OFFSET}）: "
                  f"{n_before - int((region == 'lower').sum())} 個頂點改判為上衣")
        else:
            hem_y = float(y[region == "lower"].max()) - HEM_DROP * height
            region[(region == "lower") & (y > hem_y)] = "upper"
            print(f"[INFO] 下擺（高度判定 HEM_DROP={HEM_DROP}）: "
                  f"{n_before - int((region == 'lower').sum())} 個頂點改判為上衣")

    # 裸露皮膚修正：只在厚度信號確實能區分皮膚與衣服時才有意義（見檔頭說明）。
    # （head 與 hands 共用同一個膚色，所以統一標成 hands 不影響外觀。）
    if USE_OFFSET_SKIN:
        bare_cloth = (offset < SKIN_OFFSET) & np.isin(region, ["upper", "lower"])
        n_bare = int(bare_cloth.sum())
        if n_bare > 0.25 * len(verts):
            print(f"[WARN] 裸露皮膚修正會影響 {n_bare} 個頂點（超過網格的 25%），"
                  f"這幾乎確定是門檻設錯，已跳過。")
            print(f"       請看上面的厚度分佈表重設 SKIN_OFFSET，"
                  f"或把 USE_OFFSET_SKIN 關掉。")
        else:
            region[bare_cloth] = "hands"
            print(f"[INFO] 裸露皮膚修正 (SKIN_OFFSET={SKIN_OFFSET}): "
                  f"{n_bare} 個頂點從衣服改判為皮膚")

    label = np.zeros(len(verts), np.int64)
    for name, rid in REGION_IDS.items():
        label[region == name] = rid

    # 標籤平滑：在網格鄰域上做多數決。
    # 逐頂點判定難免有零星誤判（一顆頂點被判成別的區域），平滑後邊界會走
    # 一條連續自然的路徑，而不是佈滿雜訊或沿著人工切面。
    if LABEL_SMOOTH_ITERS > 0 and not no_smooth:
        from scipy.sparse import coo_matrix
        print(f"[INFO] 標籤平滑（鄰域多數決 × {LABEL_SMOOTH_ITERS}）...")
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
        e = np.vstack([e, e[:, ::-1]])
        V = len(verts)
        A = coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])),
                       shape=(V, V)).tocsr()
        ids = sorted(REGION_IDS.values())
        for _ in range(LABEL_SMOOTH_ITERS):
            onehot = np.zeros((V, len(ids)), np.float32)
            for k, rid in enumerate(ids):
                onehot[label == rid, k] = 1.0
            votes = A @ onehot + onehot          # 鄰居票數 + 自己一票
            label = np.array(ids)[votes.argmax(axis=1)]

    for name, rid in REGION_IDS.items():
        print(f"       {name:6s} (id={rid}): {int((label == rid).sum())} 頂點")
    label = label.astype(np.float32)

    # 把頂點標籤光柵化進 UV atlas。標籤是離散值，插值會在邊界產生無意義的中間值，
    # 所以用最近整數還原 —— 邊界會有一兩個 pixel 的鋸齒，對平塗底色沒有影響。
    print("[INFO] 烘焙標籤到 UV atlas ...")
    ctx = dr.RasterizeCudaContext()
    uv_clip = np.zeros((len(uv), 4), dtype=np.float32)
    uv_clip[:, 0] = uv[:, 0] * 2.0 - 1.0
    uv_clip[:, 1] = -(uv[:, 1] * 2.0 - 1.0)
    uv_clip[:, 3] = 1.0
    faces_t = torch.tensor(faces, dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, torch.tensor(uv_clip, device="cuda").unsqueeze(0),
                           faces_t, resolution=[TEX_RES, TEX_RES])
    lab_img, _ = dr.interpolate(
        torch.tensor(label[:, None], device="cuda").unsqueeze(0), rast, faces_t)
    lab = np.rint(lab_img[0, ..., 0].detach().cpu().numpy()).astype(np.uint8)
    lab[(rast[0, ..., 3] <= 0).detach().cpu().numpy()] = 0

    np.save(work / "label_atlas.npy", lab)

    # 存一張看得懂的預覽圖：直接用各區域的平塗底色畫出來。
    # 這張圖就是 generate stage 會餵給 SD 的底色，值得打開確認邊界對不對。
    region_colors, _ = parse_prompt_colors(prompt)
    prev = np.zeros((TEX_RES, TEX_RES, 3), np.uint8)
    for name, rid in REGION_IDS.items():
        prev[lab == rid] = region_colors[name]
    cv2.imwrite(str(work / "label_preview.png"), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))

    print(f"[DONE] 語意標籤圖: {work / 'label_atlas.npy'}")
    print(f"       預覽圖（請打開確認上衣/褲子邊界）: {work / 'label_preview.png'}")
    print(f"       邊界不對就調 HEM_DROP (目前 {HEM_DROP}) 重跑這個 stage，很快。")


# ============================================================
# Stage 3: generate —— 依序生成各視角顏色並投影烘焙回 UV
# 只有這個 stage 需要 diffusers（純 numpy/cv2 + torch，不碰 nvdiffrast/xatlas/trimesh）
# ============================================================

class SDControlNetGenerator:
    """ControlNet(normal-conditioned) + SD inpaint。只在非 dry-run 時載入。"""

    def __init__(self, meta, device="cuda"):
        import torch
        from diffusers import (StableDiffusionControlNetInpaintPipeline,
                                ControlNetModel, UniPCMultistepScheduler)
        self.torch = torch
        dtype = torch.float16 if device == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(meta["controlnet_model"], torch_dtype=dtype)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            meta["sd_model"], controlnet=controlnet, torch_dtype=dtype, safety_checker=None,
        ).to(device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.device = device
        self.res = meta["sd_infer_res"]
        self.strength = meta.get("sd_strength", 0.8)
        self.steps = SD_STEPS

    def __call__(self, init_rgb, unknown_mask, normal_cond, prompt, seed=0):
        from PIL import Image
        h, w = unknown_mask.shape[:2]
        image = Image.fromarray(init_rgb.astype(np.uint8)).resize((self.res, self.res))
        mask = Image.fromarray(unknown_mask.astype(np.uint8) * 255).resize((self.res, self.res))
        control = Image.fromarray(normal_cond.astype(np.uint8)).resize((self.res, self.res))
        gen = self.torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(prompt=prompt, image=image, mask_image=mask, control_image=control,
                        num_inference_steps=self.steps, guidance_scale=7.5,
                        strength=self.strength, generator=gen).images[0]
        return np.array(out.resize((w, h)))


class DummyGenerator:
    """--dry-run 用：不載入 SD，直接回傳平塗底色。
    因為初始影像已經是語意平塗圖，dry-run 的輸出就等於「沒有布料細節的正確分區版本」，
    很適合先確認上衣/褲子邊界對不對再花 GPU 時間。
    """

    def __call__(self, init_rgb, unknown_mask, normal_cond, prompt, seed=0):
        return init_rgb.copy()


def pad_uv_borders(tex, valid_mask, iterations=6):
    """UV 島邊界顏色外擴，避免渲染時島邊界出現黑縫。"""
    tex = tex.copy()
    valid = valid_mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        dilated = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1).astype(bool)
        new_px = dilated & ~valid
        if not new_px.any():
            break
        blurred = cv2.blur(tex, (3, 3))
        tex[new_px] = blurred[new_px]
        valid = dilated
    return tex


# 第三臂消融所用的中性底色。取中等明度的無彩灰，使其不偏向任何區域的
# 目標色，且明度與各區域底色相當，避免明度差異成為額外的變因。
NEUTRAL_BASE_RGB = (128, 128, 128)


def stage_generate(work_dir, prompt, dry_run=False, sd_strength=None, recolor=0.85,
                   no_semantic=False, accumulate_full=False, no_smooth=False,
                   neutral_base=False):
    work = Path(work_dir)
    with open(work / "meta.json") as f:
        meta = json.load(f)
    tex_res = meta["tex_res"]
    if sd_strength is not None:
        meta["sd_strength"] = sd_strength
        print(f"[INFO] SD strength 覆寫為 {sd_strength}")

    atlas_mask = cv2.imread(str(work / "atlas_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    view_files = sorted((work / "views").glob("view_*.npz"))
    if not view_files:
        raise FileNotFoundError(f"{work}/views/ 裡沒有視角檔，請先跑 --stage render")

    # 語意標籤圖：有的話就用它平塗出區域底色當 SD 初始影像，
    # SD 只能在既定邊界內加布料紋理，紅色不可能跑到腿上。
    label_path = work / "label_atlas.npy"
    if no_semantic:
        # 消融用：完全不使用語意標籤，退回無約束的基準做法。
        # 這是論文中「無語意約束」對照組的正確產生方式 —— 只關掉這一個變因，
        # 其餘設定（累加策略、視角順序、色相校正）維持不變，才是乾淨的消融。
        print("[ABLATION] --no-semantic：停用語意區域約束（基準組）")
    if label_path.exists() and not no_semantic:
        label = np.load(label_path)
        # 底色從 prompt 現算，不從 meta 讀 —— 這樣換 prompt 重跑 generate 就好，
        # 不必回頭重跑 render。
        region_colors, matched = parse_prompt_colors(prompt)

        if neutral_base:
            # 消融第三臂：保留語意區域的**空間**劃分，但抽掉各區域「該是什麼
            # 顏色」的資訊，全部改為同一個中性灰。
            #
            # 其存在的理由是歸因問題：--no-semantic 同時移除了兩件事——
            # 區域的空間界定，以及區域的顏色資訊。故該組別與基準組的差異
            # 無法歸因於其中任一者。加入本臂後方構成可分離的三臂比較：
            #     A 無語意約束    ：無空間、無顏色
            #     B 中性底色      ：有空間、無顏色   ← 本臂
            #     C 基準（正確底色）：有空間、有顏色
            # A→B 的差異即為空間約束的貢獻，B→C 則為顏色資訊的貢獻。
            region_colors = {k: NEUTRAL_BASE_RGB for k in region_colors}
            matched = {}
            print(f"[ABLATION] --neutral-base：保留語意區域，"
                  f"但所有區域底色改為中性灰 {NEUTRAL_BASE_RGB}")

        base_atlas = np.zeros((tex_res, tex_res, 3), np.float32)
        for name, rid in REGION_IDS.items():
            base_atlas[label == rid] = region_colors[name]
        print(f"[INFO] 已載入語意標籤圖，會用區域底色約束生成")
        if matched:
            print(f"[INFO] 從 prompt 解析到顏色: "
                  + ", ".join(f"{r}={c}" for r, c in matched.items()))
        missing = [r for r in ["upper", "lower"] if r not in matched]
        if missing:
            # 這個失敗模式很難從結果反推：底色變灰、SD 失去分區引導，
            # 顏色會散得到處都是，看起來像方法有問題而不是 prompt 沒解析到。
            # 所以這裡講得白一點。
            print("")
            print("!" * 62)
            print(f"[WARN] prompt 裡沒解析到 {'/'.join(missing)} 的顏色！")
            print(f"       這些區域會用灰色底，SD 失去分區引導，")
            print(f"       結果很可能是顏色散到全身（臉、手都會沾到）。")
            print(f"       prompt: {prompt}")
            print(f"       請確認用了英文顏色詞 + 服裝名詞，例如：")
            print(f"         a red jacket and black pants")
            print(f"         a navy dress")
            print(f"       服裝名詞不在詞彙表裡的話，加進 GARMENT_WORDS。")
            print("!" * 62)
            print("")
    else:
        label = None
        base_atlas = np.full((tex_res, tex_res, 3), 200.0, np.float32)
        if not no_semantic:
            print(f"[WARN] 找不到 {label_path}，退回無語意約束模式。")
            print(f"       這會讓 SD 自由決定顏色畫在哪，很可能糊成一片。")
            print(f"       建議先在 chupa 環境跑 --stage semantic。")

    # 生成順序：水平的正面/背面當錨點先跑，定下整體色調，其餘視角才補縫。
    # 依序生成的視角之間沒有一致性機制，先跑面積最大的視角能減少後續漂移。
    views_meta = {v["index"]: v for v in meta.get("views", [])}

    def view_priority(vf):
        i = int(vf.stem.split("_")[1])
        vm = views_meta.get(i)
        if vm is None:
            return (2, i)
        # 仰角越接近水平、方位角越接近正/背面，優先度越高
        az_score = min(abs(np.sin(vm["azimuth"])), 1.0)
        return (abs(vm["elevation"]) + az_score, i)

    view_files = sorted(view_files, key=view_priority)
    print(f"[INFO] 讀到 {len(view_files)} 個視角，錨點視角優先")

    generator = DummyGenerator() if dry_run else SDControlNetGenerator(meta)
    print(f"[INFO] 產生器: {'DummyGenerator (--dry-run)' if dry_run else 'SDControlNetGenerator'}")

    tex_color = np.zeros((tex_res, tex_res, 3), np.float32)
    tex_weight = np.zeros((tex_res, tex_res), np.float32)

    for n, vf in enumerate(view_files):
        d = np.load(vf)
        tx, ty = d["tx"].astype(np.int64), d["ty"].astype(np.int64)
        facing = d["facing"].astype(np.float32)
        mask = d["mask"]
        view_i = int(vf.stem.split("_")[1])

        normal_img = cv2.cvtColor(
            cv2.imread(str(vf.with_name(vf.stem + "_normal.png"))), cv2.COLOR_BGR2RGB)

        # 目前累積的平均色，用來判斷每個 texel 之前有沒有內容
        with np.errstate(invalid="ignore", divide="ignore"):
            tex_avg = np.where(tex_weight[..., None] > 1e-3,
                               tex_color / np.maximum(tex_weight[..., None], 1e-6), 0.0)

        known_mask_img = mask & (tex_weight[ty, tx] > 1e-3)
        unknown_mask_img = mask & ~known_mask_img
        if not unknown_mask_img.any():
            print(f"[{n + 1}/{len(view_files)}] view {view_i}: 無新增區域，跳過生成")
            continue

        # 初始影像 = 已知區用累積結果（維持視角間連續）+ 未知區用語意底色（決定分區）
        init_img = tex_avg[ty, tx].astype(np.float32)
        init_img[~known_mask_img] = base_atlas[ty, tx][~known_mask_img]

        # 裁切到可見範圍 bbox（加 padding），提高送進 SD 的解析度利用率
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = int(0.05 * max(y1 - y0, x1 - x0))
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        y1 = min(mask.shape[0], y1 + pad)
        x1 = min(mask.shape[1], x1 + pad)

        gen_crop = generator(init_img[y0:y1, x0:x1],
                             unknown_mask_img[y0:y1, x0:x1],
                             normal_img[y0:y1, x0:x1], prompt, seed=view_i)

        gen_full = init_img.copy()
        gen_full[y0:y1, x0:x1] = gen_crop.astype(np.float32)

        # 關鍵設計：只累加「這個視角新生成的」像素。
        # 若把整張圖累加回去，已畫好的區域每個視角都會被 SD 重新編解碼一次，
        # 50 個視角下來會被洗成灰白糊狀。已知區域一旦定下就不再更動。
        # --accumulate-full 用於論文的累加策略消融，重現舊做法。
        acc = mask if accumulate_full else unknown_mask_img
        w = (facing[acc] ** 2.0).astype(np.float32)
        np.add.at(tex_color, (ty[acc], tx[acc]), gen_full[acc] * w[:, None])
        np.add.at(tex_weight, (ty[acc], tx[acc]), w)

        print(f"[{n + 1}/{len(view_files)}] view {view_i}: "
              f"可見 {int(mask.sum())} px, 新增 {int(acc.sum())} px")

    valid = tex_weight > 1e-3
    tex = np.zeros_like(tex_color)
    tex[valid] = tex_color[valid] / tex_weight[valid, None]
    tex = np.clip(tex, 0, 255).astype(np.uint8)

    # 色相校正：把每個區域的顏色拉回它應有的底色，但保留 SD 生成的明暗細節。
    #
    # 即使給了區域底色當起點，SD 在 strength 較高或視角較少時仍會讓顏色暈開，
    # 兩種顏色的服裝（例如藍外套配卡其褲）會互相污染，看起來像迷彩。
    # 這裡在 LAB 空間只改色度通道 (a,b)、保留亮度通道 (L)，
    # 所以布料的皺褶與陰影都還在，只是顏色被拉回正確的區域色。
    if recolor > 0 and label is not None:
        lab_img = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(np.float32)
        n_fixed = 0
        for name, rid in REGION_IDS.items():
            m = label == rid
            if not m.any():
                continue
            t = cv2.cvtColor(np.array(region_colors[name], np.uint8).reshape(1, 1, 3),
                             cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
            lab_img[m, 1] = lab_img[m, 1] * (1 - recolor) + t[1] * recolor
            lab_img[m, 2] = lab_img[m, 2] * (1 - recolor) + t[2] * recolor
            n_fixed += int(m.sum())
        tex = cv2.cvtColor(np.clip(lab_img, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        print(f"[INFO] 色相校正（強度 {recolor}）：{n_fixed} texel 的顏色拉回區域底色，"
              f"明暗細節保留")

    real_pct = 100.0 * (valid & atlas_mask).sum() / max(atlas_mask.sum(), 1)
    raw_pct = 100.0 * valid.sum() / valid.size
    print(f"\n[INFO] 覆蓋率（相對整張貼圖）: {raw_pct:.1f}%")
    print(f"[INFO] 覆蓋率（相對 UV atlas 有效面積，這個才是真實表面覆蓋率）: {real_pct:.1f}%")

    # 診斷圖：綠=多視角直接看到、紅=有表面但沒被任何視角看到(要靠inpaint補)、黑=打包空隙
    dbg = np.zeros((tex_res, tex_res, 3), np.uint8)
    dbg[atlas_mask & valid] = (0, 200, 0)
    dbg[atlas_mask & ~valid] = (0, 0, 220)
    cv2.imwrite(str(work / "coverage_debug.png"), dbg)
    print(f"[INFO] 覆蓋率診斷圖: {work / 'coverage_debug.png'}")

    missing = (~valid).astype(np.uint8) * 255
    if missing.any():
        # 有語意標籤時，先把沒看到的區域填上它自己的區域底色再 inpaint。
        # 否則 cv2.inpaint 會從鄰近區域抓色，腋下可能被塗成褲子的黑、腳底被塗成膚色。
        if label is not None:
            tex[~valid] = base_atlas[~valid].astype(np.uint8)
            print("[INFO] 未覆蓋區域先填區域底色，再做 inpainting 平滑 ...")
        else:
            print("[INFO] 用傳統 inpainting 補完全沒被看到的區域（腋下/腳底等）...")
        tex = cv2.inpaint(tex, missing, 5, cv2.INPAINT_TELEA)

    print("[INFO] UV 島邊界顏色外擴 ...")
    tex = pad_uv_borders(tex, valid, iterations=6)

    cv2.imwrite(str(work / "texture.png"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    print(f"[DONE] generate stage 完成，貼圖: {work / 'texture.png'}")
    return tex


# ============================================================
# Stage 3: export —— 輸出 obj + mtl + texture
# 純檔案寫入，哪個環境都能跑
# ============================================================

def stage_export(work_dir, out_dir, name="chupa_textured"):
    work = Path(work_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    g = np.load(work / "geometry.npz")
    verts, faces, uv = g["verts"], g["faces"], g["uv"]
    tex_src = work / "texture.png"
    if not tex_src.exists():
        raise FileNotFoundError(f"找不到 {tex_src}，請先跑 --stage generate")

    obj_path, mtl_path = out / f"{name}.obj", out / f"{name}.mtl"
    tex_path = out / f"{name}_texture.png"
    cv2.imwrite(str(tex_path), cv2.imread(str(tex_src)))

    with open(mtl_path, "w") as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
        f.write(f"map_Kd {tex_path.name}\n")

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_path.name}\nusemtl material_0\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for c in uv:
            # 已知問題：這份 OBJ 在 Blender 中貼圖會錯位，且兩個 V 軸方向都不對，
            # 所以不是單純的翻轉問題。推測與 Blender 的 OBJ 匯入器合併重複頂點
            # 有關 —— xatlas 會在 UV 接縫處複製頂點（位置相同、UV 不同），
            # 一旦被合併，該處的 UV 對應就會壞掉。尚未確認。
            #
            # 要檢視結果請用下面同時輸出的 .glb（trimesh 匯出，已驗證正確）。
            # OBJ 保留是為了與只吃 OBJ 的工具相容。
            vv = (1.0 - c[1]) if OBJ_FLIP_V else c[1]
            f.write(f"vt {c[0]:.6f} {vv:.6f}\n")
        for tri in faces:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    # 同時輸出 glb。這是建議用來檢視結果的檔案 —— trimesh 會自己處理 glTF 的
    # UV 慣例與頂點複製，實測正確；而且貼圖直接嵌在檔案裡，不會有找不到
    # 貼圖或相對路徑的問題。
    glb_path = out / f"{name}.glb"
    try:
        import trimesh
        from trimesh.visual import TextureVisuals
        from PIL import Image
        m = trimesh.Trimesh(
            vertices=verts, faces=faces, process=False,
            visual=TextureVisuals(uv=uv, image=Image.open(tex_src).convert("RGB")))
        m.export(str(glb_path))
    except Exception as e:
        glb_path = None
        print(f"[WARN] glb 輸出失敗（{e}），只會有 obj。"
              f"obj 在 Blender 中貼圖會錯位，這是已知問題。")

    if glb_path:
        print(f"[DONE] 檢視用（建議）: {glb_path}")
        print(f"       Blender: File → Import → glTF 2.0")
    print(f"[DONE] 相容用 obj: {obj_path}")
    print(f"       （注意：obj 在 Blender 中貼圖會錯位，已知問題，請用上面的 glb）")
    print(f"       貼圖: {tex_path}")
    return glb_path or obj_path


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["render", "semantic", "generate", "export"])
    p.add_argument("--mesh", default=CHUPA_MESH, help="render stage: mesh_refine.obj 路徑")
    p.add_argument("--work", default=WORK_DIR, help="stage 之間的交換資料夾")
    p.add_argument("--out", default=OUT_DIR, help="export stage: 最終輸出資料夾")
    p.add_argument("--prompt", default=PROMPT, help="建議跟生成幾何時同一句")
    p.add_argument("--smplx-obj", default=SMPLX_SOURCE_OBJ,
                   help="semantic stage: 對齊到 chupa mesh、且維持 SMPL-X 原始拓樸的 obj")
    p.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR,
                   help="semantic stage: SMPL-X 模型資料夾")
    p.add_argument("--dry-run", action="store_true",
                   help="generate stage: 不載入 SD，直接輸出平塗底色，"
                        "用來確認上衣/褲子邊界對不對")
    p.add_argument("--sd-strength", type=float, default=None,
                   help="SD 對區域底色的改動幅度（預設 0.8）。"
                        "兩種顏色的服裝互相污染時調低到 0.5~0.6")
    p.add_argument("--recolor", type=float, default=0.85,
                   help="色相校正強度 0~1（預設 0.85）。在 LAB 空間把各區域的"
                        "色度拉回底色、保留明暗細節。設 0 可關閉")
    # ── 論文消融用的旗標 ──
    p.add_argument("--no-semantic", action="store_true",
                   help="[消融] 停用語意區域約束，退回無約束基準（無空間、無顏色）")
    p.add_argument("--neutral-base", action="store_true",
                   help="[消融] 保留語意區域的空間劃分，但各區域底色改為中性灰"
                        "（有空間、無顏色）。與 --no-semantic 及基準組並列，"
                        "方可分離空間約束與顏色資訊各自的貢獻")
    p.add_argument("--accumulate-full", action="store_true",
                   help="[消融] 累加完整影像而非僅新生成像素，重現舊策略")
    p.add_argument("--no-smooth", action="store_true",
                   help="[消融] semantic stage: 停用標籤平滑")
    args = p.parse_args()

    if args.stage == "render":
        stage_render(args.mesh, args.work)
    elif args.stage == "semantic":
        stage_semantic(args.work, args.smplx_obj, args.smplx_model_dir, args.prompt,
                       no_smooth=args.no_smooth)
    elif args.stage == "generate":
        stage_generate(args.work, args.prompt, dry_run=args.dry_run,
                       sd_strength=args.sd_strength, recolor=args.recolor,
                       no_semantic=args.no_semantic,
                       accumulate_full=args.accumulate_full,
                       neutral_base=args.neutral_base)
        print("\n[INFO] 自動接著跑 export ...")
        stage_export(args.work, args.out)
    else:
        stage_export(args.work, args.out)


if __name__ == "__main__":
    main()