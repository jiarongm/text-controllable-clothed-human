r"""
make_fig48.py —— 產生論文圖 4-8（不同體型的著衣網格驅動結果）

圖說宣稱四件事，**每一件都無法由圖面確認**，故本腳本逐項核對後才併圖：

  一、三者體型不同          → 以包圍盒高度與 SMPL-X 形狀參數核對
  二、穿同一件服裝          → 比對各資產 meta.json 中的提示詞字串，須完全相同
  三、由同一段動作序列驅動  → 三者共用同一段、同一影格的姿態（結構上保證）
  四、腳底對齊至同一平面    → 平移各網格使最低點等高

**第二、三項是整張圖的論證前提**：組內差異須純粹來自體型。若三者用的是不同
的生成執行或不同的動作片段，此圖即不構成對照，而這從圖面完全看不出來。

一個容易忽略的陷阱
==================
若各格各自依自身包圍盒取景，**三種體型會被縮放成一樣高**，身高差異完全消失
——而該差異正是本圖要呈現的。故本腳本以三者的聯集包圍盒計算**共用取景**，
各格不得各自縮放。此與圖 4-15 各格須共用取景為同一問題。

用法
====
  # 先看有哪些體型資產、提示詞是否一致
  python make_fig48.py --list

  # 併圖（預設取同一身高、三種體重的一組）
  python make_fig48.py --out figures

  # 指定三組
  python make_fig48.py --assets B1_h165_w52,B2_h165_w60,B3_h165_w74 --out figures

環境：chupa（需 torch、nvdiffrast、opencv）
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo_work"
SMPLX_DIR = ROOT / "data" / "renderpeople" / "smplx"

# 預設三組：同一身高、三種體重，故差異純為胖瘦。
# 若改用不同身高的組合，圖說須一併改寫（身高與胖瘦為兩個因子）。
DEFAULT_ASSETS = ["B1_h165_w52", "B2_h165_w60", "B3_h165_w74"]

CANVAS = 512


def parse_meta(work):
    """自 meta.json 的 mesh_path 取出受測體型（subject）與提示詞。

    提示詞與 subject 皆編碼於生成輸出的目錄結構中：
      .../<時間戳>/<subject>/<提示詞>_<種子>/mesh_refine.obj
    """
    mp = work / "meta.json"
    if not mp.exists():
        return None
    try:
        mesh_path = json.loads(mp.read_text(encoding="utf-8"))["mesh_path"]
    except Exception:
        return None
    # parts[-1] 為 mesh_refine.obj 本身，故自末端往回數：
    #   [-2] 提示詞目錄、[-3] 體型 subject、[-4] 執行時間戳
    parts = Path(mesh_path).parts
    if len(parts) < 4:
        return None
    prompt_dir, subject, stamp = parts[-2], parts[-3], parts[-4]
    # 尾端的 _0042 為種子，屬於執行而非服裝，比對提示詞時須剝除
    prompt = re.sub(r"_\d+$", "", prompt_dir)
    return {"subject": subject, "prompt": prompt, "seed_dir": prompt_dir,
            "stamp": stamp, "mesh_path": mesh_path}


def bind_path_for(subject):
    p = SMPLX_DIR / f"{subject}.pkl"
    return p if p.exists() else None


def scan():
    out = []
    if not DEMO.is_dir():
        sys.exit(f"[錯誤] 找不到 {DEMO}")
    for d in sorted(DEMO.glob("B*")):
        if not d.is_dir():
            continue
        m = parse_meta(d)
        rec = {"name": d.name, "dir": d, "meta": m,
               "has_geo": (d / "geometry.npz").exists(),
               "has_sw": (d / "skin_weights.npz").exists(),
               "has_tex": (d / "texture.png").exists()}
        rec["bind"] = bind_path_for(m["subject"]) if m else None
        out.append(rec)
    return out


def show(rows):
    print(f"\n{'資產':<18}{'體型 subject':<26}{'幾何':>5}{'權重':>5}"
          f"{'貼圖':>5}{'綁定':>5}  提示詞")
    for r in rows:
        m = r["meta"] or {}
        print(f"{r['name']:<18}{m.get('subject', '—'):<26}"
              f"{'✓' if r['has_geo'] else '✗':>5}"
              f"{'✓' if r['has_sw'] else '✗':>5}"
              f"{'✓' if r['has_tex'] else '✗':>5}"
              f"{'✓' if r['bind'] else '✗':>5}  {m.get('prompt', '—')}")

    prompts = {}
    for r in rows:
        if r["meta"]:
            prompts.setdefault(r["meta"]["prompt"], []).append(r["name"])
    print("\n依提示詞分組（同一組方可併入圖 4-8）：")
    for p, names in sorted(prompts.items(), key=lambda kv: -len(kv[1])):
        print(f"  [{len(names)} 筆] {p}")
        print(f"        {'、'.join(names)}")
    print("\n提示：圖 4-8 之圖說稱三者「穿著同一件服裝」，"
          "故三者必須落在同一組。")


# 貼圖候選，依優先順序。**務必優先取已修補的最終貼圖**——
# texture.png 為烘焙所得的原始圖集，各 chart 邊界未填補，渲染會呈椒鹽狀雜點。
TEX_CANDIDATES = [
    "final/chupa_textured_texture.png",
    "chupa_textured_texture.png",
    "texture.png",
]


def load_asset(work, need_tex=True):
    import cv2
    g = np.load(work / "geometry.npz")
    sw = np.load(work / "skin_weights.npz")
    tex = None
    if need_tex:
        for rel in TEX_CANDIDATES:
            tp = work / rel
            if tp.exists():
                if rel == "texture.png":
                    print(f"  [警告] {work.name} 僅有未修補的原始圖集 "
                          f"texture.png，貼圖渲染會出現椒鹽狀雜點")
                tex = cv2.cvtColor(cv2.imread(str(tp)),
                                   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                break
    return {"verts": g["verts"].astype(np.float64),
            "faces": g["faces"].astype(np.int32),
            "uv": g["uv"].astype(np.float32),
            "sj": sw["joints"].astype(np.int64),
            "sw": sw["weights"].astype(np.float64),
            "tex": tex}


def drive(asset, bind, J_rest, parents, full_pose):
    from drive_avatar import global_transforms
    G = global_transforms(J_rest, full_pose, parents)
    G_bi = np.linalg.inv(global_transforms(J_rest, bind["full_pose"], parents))
    T = np.einsum("jab,jbc->jac", G, G_bi)
    v_model = (asset["verts"] - bind["transl"].reshape(1, 3)) / bind["scale"]
    vh = np.concatenate([v_model, np.ones((len(v_model), 1))], axis=1)
    M = np.einsum("nk,nkab->nab", asset["sw"], T[asset["sj"]])
    return np.einsum("nab,nb->na", M, vh)[:, :3]


def vertex_normals(v, f):
    n = np.zeros_like(v, dtype=np.float64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-12)


def render(verts, asset, frame, size=CANVAS, flip_y=True, mode="normal"):
    r"""以**外部傳入**的取景 frame=(cx, cy, half) 渲染，各格不得自行縮放。

    mode="normal"（預設）以頂點法向量著色，不使用貼圖。本圖所主張者為
    **幾何**（體型輪廓），貼圖只會將圖集接縫的雜訊疊在輪廓上，妨礙判讀；
    且 B 系列資產未走完貼圖修補流程（無 final/ 目錄），其 texture.png 為
    未修補之原始圖集，渲染結果呈椒鹽狀雜點。
    """
    import torch
    import nvdiffrast.torch as dr

    v = verts.astype(np.float32)
    cx, cy, half = frame
    clip = np.ones((len(v), 4), np.float32)
    clip[:, 0] = (v[:, 0] - cx) / half
    clip[:, 1] = (v[:, 1] - cy) / half
    # 深度符號：SMPL-X 人體朝向 +Z，正面頂點世界 z 較大；nvdiffrast 保留
    # NDC z 較小者，故須取負號，否則會渲出背面（此錯誤曾發生於圖 4-15）。
    clip[:, 2] = np.clip(-(v[:, 2] - v[:, 2].mean()) / (half * 4), -0.9, 0.9)

    dev = "cuda"
    ctx = dr.RasterizeCudaContext()
    clip_t = torch.tensor(clip, device=dev).unsqueeze(0)
    faces_t = torch.tensor(asset["faces"], dtype=torch.int32, device=dev)
    rast, _ = dr.rasterize(ctx, clip_t, faces_t, resolution=[size, size])
    vis = (rast[..., 3:4] > 0).float()

    if mode == "texture" and asset["tex"] is not None:
        uv = asset["uv"].copy()
        uv[:, 1] = 1.0 - uv[:, 1]          # 附錄 A.2
        uvi, _ = dr.interpolate(torch.tensor(uv, device=dev).unsqueeze(0),
                                rast, faces_t)
        col = dr.texture(torch.tensor(asset["tex"], device=dev).unsqueeze(0),
                         uvi, filter_mode="linear")
    else:
        nrm = vertex_normals(verts, asset["faces"].astype(np.int64))
        nt = torch.tensor(nrm.astype(np.float32), device=dev).unsqueeze(0)
        ni, _ = dr.interpolate(nt, rast, faces_t)
        ni = torch.nn.functional.normalize(ni, dim=-1)
        # 正面打光：視線方向 +z（因上方已將 z 取負，此處以 -n_z 為朝向觀者）
        lam = torch.clamp(-ni[..., 2:3], 0.0, 1.0)
        shade = 0.30 + 0.70 * lam
        col = shade.repeat(1, 1, 1, 3)

    rgb = (col * vis + (1.0 - vis))[0].detach().cpu().numpy()   # 背景為白
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if flip_y:
        img = np.flipud(img)
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-8")
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    ap.add_argument("--out", default="figures")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--temporal", default="temporal_eval")
    ap.add_argument("--motion", default="shoulder")
    ap.add_argument("--frame", type=int, default=-1,
                    help="姿態取自該段的第幾影格；預設取中段")
    ap.add_argument("--model-dir", default="src/smpl_related/models")
    ap.add_argument("--shading", choices=["normal", "texture"], default="normal",
                    help="預設以法向量著色。本圖主張的是幾何，不宜貼圖")
    ap.add_argument("--allow-mismatch", action="store_true")
    ap.add_argument("--no-flip-y", action="store_true")
    args = ap.parse_args()

    rows = scan()
    if args.list:
        show(rows)
        return

    import cv2
    from drive_avatar import load_bind, build_smplx, motion_to_full_pose

    by_name = {r["name"]: r for r in rows}
    want = [s.strip() for s in args.assets.split(",") if s.strip()]
    picked = []
    for n in want:
        r = by_name.get(n)
        if r is None:
            sys.exit(f"[錯誤] 找不到資產 {n}。請執行 --list 檢視。")
        for k, lbl in (("has_geo", "geometry.npz"), ("has_sw", "skin_weights.npz")):
            if not r[k]:
                sys.exit(f"[錯誤] {n} 缺少 {lbl}")
        if r["meta"] is None:
            sys.exit(f"[錯誤] {n} 的 meta.json 無法解析，無從核對服裝是否相同")
        if r["bind"] is None:
            sys.exit(f"[錯誤] 找不到 {n} 之綁定檔 "
                     f"{SMPLX_DIR / (r['meta']['subject'] + '.pkl')}")
        picked.append(r)

    # ---- 條件二：同一件服裝 ----
    prompts = {r["meta"]["prompt"] for r in picked}
    print("\n各資產之生成來源：")
    for r in picked:
        m = r["meta"]
        print(f"  {r['name']:<18}{m['subject']:<26}{m['stamp']}")
        print(f"      提示詞：{m['prompt']}")
    if len(prompts) == 1:
        print(f"\n  ✓ 三者提示詞相同：{prompts.pop()}")
    else:
        print(f"\n  ✗ **提示詞不一致**：{sorted(prompts)}")
        print("     圖說「穿著同一件服裝」與事實不符。"
              "須改用同一提示詞的資產，或修正圖說。")
        if not args.allow_mismatch:
            sys.exit(1)

    # ---- 條件三：同一段、同一影格的姿態 ----
    pp = Path(args.temporal) / f"pose_{args.motion}.npz"
    if not pp.exists():
        sys.exit(f"[錯誤] 找不到 {pp}，請先執行 run_romp_batch.py")
    poses = np.load(pp)["poses"]
    fi = args.frame if args.frame >= 0 else len(poses) // 2
    if not (0 <= fi < len(poses)):
        sys.exit(f"[錯誤] --frame {fi} 超出範圍（該段共 {len(poses)} 影格）")
    print(f"\n  ✓ 三者共用 {args.motion} 段第 {fi} 影格之姿態"
          f"（該段共 {len(poses)} 影格）")

    # ---- 驅動 ----
    driven, assets, labels, bind_h = [], [], [], []
    for r in picked:
        a = load_asset(r["dir"])
        if args.shading == "texture" and a["tex"] is None:
            print(f"  [警告] {r['name']} 無 texture.png")
        bind = load_bind(str(r["bind"]))
        J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
        full = motion_to_full_pose(poses, bind["full_pose"])
        v = drive(a, bind, J_rest, parents, full[fi])
        driven.append(v)
        assets.append(a)
        labels.append(r["name"].split("_", 1)[1])   # h165_w52
        # 綁定姿勢的身高須**換算至模型座標**後再量。資產的 verts 存於世界
        # 座標（已乘 bind scale、加 transl），而 drive() 回傳的是模型座標；
        # 兩者直接相比即為不同座標系的混用，會憑空產生約一倍的差距。
        vm = (a["verts"] - bind["transl"].reshape(1, 3)) / bind["scale"]
        bind_h.append(float(vm[:, 1].max() - vm[:, 1].min()))

    # ---- 條件四：腳底對齊至同一平面 ----
    floor = min(v[:, 1].min() for v in driven)
    driven = [v - np.array([0.0, v[:, 1].min() - floor, 0.0]) for v in driven]

    # ---- 條件一：體型確實不同 ----
    #
    # 身高**須於綁定姿勢量測**。驅動後的包圍盒高度並非身高——若該影格的
    # 動作為手臂高舉，包圍盒上緣即為指尖而非頭頂，量到的數值會比身高
    # 大上三十公分，且隨動作而變，無從與資產命名中的身高對照。
    print("\n量測（模型座標，單位：公尺）：")
    print(f"{'資產':<12}{'綁定姿勢身高':>14}{'驅動後最大寬度':>16}"
          f"{'驅動後包圍盒高':>16}")
    heights = list(bind_h)
    for hb, v, lb in zip(bind_h, driven, labels):
        w = float(v[:, 0].max() - v[:, 0].min())
        hd = float(v[:, 1].max() - v[:, 1].min())
        print(f"{lb:<12}{hb:>14.4f}{w:>16.4f}{hd:>16.4f}")
    print("  （末欄含手臂等肢體於該姿態下的伸展，非身高，僅供除錯參考）")

    named = [l.split("_")[0] for l in labels]        # h165 等
    if len(set(named)) == 1:
        print(f"\n  ※ 三者身高相同（{named[0]}），差異純為體重。"
              "\n     圖說若稱「三種體型」，宜明寫其為同一身高下的三種體重，"
              "否則讀者無從得知變動的是哪一個因子。")
    elif max(heights) - min(heights) < 1e-3:
        print("\n  [警告] 資產命名所示的身高不同，實測綁定姿勢身高卻幾乎相同"
              "——請確認選用的資產是否正確。")

    # ---- 共用取景（關鍵：不可各格自行縮放）----
    #
    # 垂直方向（cy 與 half）必須三格共用，否則各體型會被縮放成等高，
    # 身高差異消失。水平位置（cx）則可逐格置中——人在畫面中的左右位置
    # 不承載任何體型資訊，逐格置中只是讓版面整齊。
    allv = np.concatenate(driven, axis=0)
    cy = (allv[:, 1].max() + allv[:, 1].min()) / 2
    half = allv[:, 1].ptp() * 0.56
    print(f"\n  ✓ 三格共用垂直取景（半視野 {half:.4f} 公尺），"
          f"故高矮胖瘦於圖面上為真實比例")

    imgs = []
    for v, a in zip(driven, assets):
        cx = (v[:, 0].max() + v[:, 0].min()) / 2
        im = render(v, a, (cx, cy, half),
                    flip_y=not args.no_flip_y, mode=args.shading)
        imgs.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    band = 36
    tiles = []
    for im, lb in zip(imgs, labels):
        b = np.full((band, im.shape[1], 3), 255, np.uint8)
        cv2.putText(b, lb, (8, band - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, b]))
    grid = np.hstack(tiles)

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    # 檔名帶入著色方式，否則兩種模式會互相覆蓋（曾發生：先產法向量版，
    # 再跑貼圖版，結果只剩下貼圖版）
    suffix = "" if args.shading == "normal" else f"_{args.shading}"
    p = od / f"fig_4-8_bodyshapes{suffix}.png"
    cv2.imwrite(str(p), grid)
    print(f"\n[完成] {p}")
    print("\n圖說核對：提示詞為上列所示者，"
          "請確認論文圖說對服裝的描述與其一致"
          "（例如提示詞若為 white t-shirt and blue jeans，"
          "圖說不宜寫成「白色上衣與長褲」以外的說法）。")


if __name__ == "__main__":
    main()
