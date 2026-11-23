r"""
make_fig413.py —— 產生論文圖 4-13（最終著衣化身於動作序列中數個影格的呈現）

圖 4-13 置於 4.5 節末，其作用為呈現外觀生成流程的最終產物：帶貼圖、且可被
動作驅動的著衣化身。

與圖 4-15 的分工
================
兩圖內容相近（皆為貼圖後之化身隨動作變化），惟所屬章節的論點不同：
  圖 4-13（4.5 節末）—— 外觀生成的產物為何
  圖 4-15（4.8.1 節）—— 驅動是否正確跟隨輸入影片（故另有輸入影格一排）
為避免讀者誤以為重複排版錯誤，本腳本預設採用**與圖 4-15 不同的資產與動作**，
並以較多影格呈現。兩圖所用之資產與動作段會印出，供圖說據實標示。

貼圖的選擇
==========
資產目錄下可能同時存在兩張貼圖：
  texture.png                  —— 烘焙所得之**原始圖集，未經修補**
  chupa_textured_texture.png   —— 走完修補流程之**最終貼圖**
務必使用後者。以未修補的原始圖集渲染，畫面會呈椒鹽狀雜點（圖集各 chart
邊界漏取到未填補區域所致）——此情形曾實際發生於圖 4-8 的初版。
本腳本自動優先選用最終貼圖，若僅存原始圖集則明確警告。

用法
====
  python make_fig413.py --list                       # 看哪些資產有最終貼圖
  python make_fig413.py --out figures
  python make_fig413.py --work demo_work/A1_red_dress_v2 --motion run --cols 6

環境：chupa（需 torch、nvdiffrast、opencv）
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# 圖 4-15 所用者（見 make_demo_figures.py），此處刻意避開
FIG415_WORK, FIG415_MOTION = "demo_work/A2_navy_coat", "shoulder"

DEFAULT_WORK = "demo_work/A2_navy_coat"      # 兩件式：海軍藍外套配卡其褲
DEFAULT_MOTION = "run"

# 4.5.6 節的結論為**品質組態（50 視角）應作為最終資產的預設配置**。
# 圖 4-13 呈現的是「最終產物」，若採快速組態（12 視角）即與該結論相左，
# 且讀者無從由圖面察覺。故此處檢查資產的視角數並明確警告。
QUALITY_VIEWS = 50

CANVAS = 512
N_COLS = 6

# 最終貼圖的候選檔名，依優先順序。排在前者為走完修補流程之輸出。
TEX_CANDIDATES = [
    "final/chupa_textured_texture.png",
    "chupa_textured_texture.png",
    "texture.png",                      # 原始圖集，未修補——最後才用
]
RAW_TEX = "texture.png"


def find_texture(work):
    for rel in TEX_CANDIDATES:
        p = work / rel
        if p.exists():
            return p, (rel == RAW_TEX)
    return None, False


def read_meta(d):
    import json
    p = d / "meta.json"
    if not p.exists():
        return None, None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    mp = j.get("mesh_path", "")
    parts = Path(mp).parts
    prompt = re.sub(r"_\d+$", "", parts[-2]) if len(parts) >= 2 else None
    return j.get("n_views"), prompt


def list_assets(root):
    print(f"\n{'資產':<28}{'視角':>5}{'幾何':>5}{'權重':>5}{'貼圖':>10}  提示詞")
    for d in sorted(root.glob("*")):
        if not d.is_dir():
            continue
        tex, raw = find_texture(d)
        if tex is None and not (d / "geometry.npz").exists():
            continue
        nv, prompt = read_meta(d)
        tmark = "—" if tex is None else ("原始圖集" if raw else "最終貼圖")
        print(f"{str(d.relative_to(ROOT)):<28}"
              f"{str(nv if nv is not None else '—'):>5}"
              f"{'✓' if (d / 'geometry.npz').exists() else '✗':>5}"
              f"{'✓' if (d / 'skin_weights.npz').exists() else '✗':>5}"
              f"{tmark:>10}  {prompt or '—'}")
    print("\n選用準則有二，缺一不可：")
    print(f"  1. 貼圖須為「最終貼圖」——以原始圖集渲染會出現椒鹽狀雜點。")
    print(f"  2. 視角須為 {QUALITY_VIEWS}（品質組態）——4.5.6 節的結論為品質"
          "組態應作為最終資產的預設配置，"
          f"\n     以 12 視角的資產充當「最終產物」即與該結論相左。")
    print("\n若所需的兩件式資產僅有 12 視角，可僅重跑貼圖烘焙（幾何不需重生成）："
          "\n  unset UVBAKE_FAST && python build_asset.py --work <資產目錄>"
          "\n依表 4-17，該階段於品質組態下約需 154 秒。")


def load_asset(work, tex_path):
    import cv2
    g = np.load(work / "geometry.npz")
    sw = np.load(work / "skin_weights.npz")
    tex = cv2.cvtColor(cv2.imread(str(tex_path)),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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


def render(verts, asset, frame, size=CANVAS, flip_y=True):
    """frame=(cx, cy, half) 由呼叫端傳入；各格共用，不得自行縮放。"""
    import torch
    import nvdiffrast.torch as dr

    v = verts.astype(np.float32)
    cx, cy, half = frame
    clip = np.ones((len(v), 4), np.float32)
    clip[:, 0] = (v[:, 0] - cx) / half
    clip[:, 1] = (v[:, 1] - cy) / half
    # 深度取負：SMPL-X 人體朝 +Z，而 nvdiffrast 保留 NDC z 較小者
    clip[:, 2] = np.clip(-(v[:, 2] - v[:, 2].mean()) / (half * 4), -0.9, 0.9)

    dev = "cuda"
    ctx = dr.RasterizeCudaContext()
    clip_t = torch.tensor(clip, device=dev).unsqueeze(0)
    faces_t = torch.tensor(asset["faces"], dtype=torch.int32, device=dev)
    uv = asset["uv"].copy()
    uv[:, 1] = 1.0 - uv[:, 1]            # 附錄 A.2
    rast, _ = dr.rasterize(ctx, clip_t, faces_t, resolution=[size, size])
    uvi, _ = dr.interpolate(torch.tensor(uv, device=dev).unsqueeze(0),
                            rast, faces_t)
    col = dr.texture(torch.tensor(asset["tex"], device=dev).unsqueeze(0),
                     uvi, filter_mode="linear")
    vis = (rast[..., 3:4] > 0).float()
    rgb = (col * vis + (1.0 - vis))[0].detach().cpu().numpy()
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if flip_y:
        img = np.flipud(img)
    return np.ascontiguousarray(img)


def main():
    ap = argparse.ArgumentParser(description="產生圖 4-13")
    ap.add_argument("--work", default=DEFAULT_WORK)
    ap.add_argument("--demo-root", default="demo_work")
    ap.add_argument("--bind", default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    ap.add_argument("--model-dir", default="src/smpl_related/models")
    ap.add_argument("--temporal", default="temporal_eval")
    ap.add_argument("--motion", default=DEFAULT_MOTION)
    ap.add_argument("--cols", type=int, default=N_COLS)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--allow-raw-texture", action="store_true")
    ap.add_argument("--no-flip-y", action="store_true")
    args = ap.parse_args()

    if args.list:
        list_assets(ROOT / args.demo_root)
        return

    import cv2
    from drive_avatar import load_bind, build_smplx, motion_to_full_pose

    work = Path(args.work)
    if not work.is_absolute():
        work = ROOT / work
    if not (work / "geometry.npz").exists():
        sys.exit(f"[錯誤] {work} 缺少 geometry.npz。以 --list 檢視可用資產。")

    tex, raw = find_texture(work)
    if tex is None:
        sys.exit(f"[錯誤] {work} 下找不到任何貼圖")
    if raw:
        print(f"[警告] {work.name} 僅有未修補的原始圖集 {RAW_TEX}，"
              "\n       渲染結果將呈椒鹽狀雜點，不適合作為「最終產物」的展示。"
              "\n       請改用已走完修補流程的資產（--list 檢視），"
              "或加 --allow-raw-texture 強制。")
        if not args.allow_raw_texture:
            sys.exit(1)
    print(f"[INFO] 貼圖：{tex.relative_to(work)}"
          f"（{'原始圖集' if raw else '最終貼圖'}）")

    # 貼圖與幾何須出自同一次烘焙。若 final/ 下留著舊的貼圖，而 geometry.npz
    # 已被新的一次生成覆蓋，二者的 UV 對不上，渲染結果是把 A 的貼圖貼到 B 的
    # 網格上——畫面看起來仍像一件衣服，從圖面無從察覺。
    # 此情形實際發生過（A2_navy_coat 被一次紅色禮服的生成覆蓋）。
    geo_t = (work / "geometry.npz").stat().st_mtime
    tex_t = tex.stat().st_mtime
    if tex_t < geo_t - 60:
        import datetime as _dt
        f = "%Y-%m-%d %H:%M"
        sys.exit(
            f"\n[錯誤] 貼圖的時間早於幾何：\n"
            f"       geometry.npz  {_dt.datetime.fromtimestamp(geo_t):{f}}\n"
            f"       {tex.relative_to(work)}  "
            f"{_dt.datetime.fromtimestamp(tex_t):{f}}\n"
            "       兩者應出自同一次烘焙。貼圖較舊，表示幾何已被重新產生而"
            "貼圖未更新，\n       渲染會把舊貼圖貼到新網格上，UV 完全對不上。\n"
            "       請重跑 export 步驟，或確認本目錄的內容是否已被其他生成覆蓋。")

    nv, prompt = read_meta(work)
    print(f"[INFO] 提示詞：{prompt or '（無法自 meta.json 取得）'}")
    print(f"[INFO] 視角數：{nv if nv is not None else '—'}")
    if nv is not None and nv != QUALITY_VIEWS:
        print(f"\n[警告] 本資產為 {nv} 視角（快速組態），非品質組態"
              f"（{QUALITY_VIEWS} 視角）。"
              "\n       4.5.6 節的結論為品質組態應作為最終資產的預設配置；"
              "\n       以快速組態的資產充當圖 4-13 的「最終產物」與該結論相左，"
              "\n       且讀者無從由圖面察覺（此與圖 4-5 的參數不一致同類）。"
              "\n       處理方式二擇一："
              "\n         甲、僅重跑貼圖烘焙（幾何不需重生成，約 154 秒）："
              "\n             unset UVBAKE_FAST && python build_asset.py "
              f"--work {args.work}"
              "\n         乙、沿用本資產，但圖說須據實標明其為 "
              f"{nv} 視角之快速組態。\n")

    if str(Path(args.work)) == FIG415_WORK and args.motion == FIG415_MOTION:
        print(f"[警告] 所選資產與動作與圖 4-15 完全相同"
              f"（{FIG415_WORK} / {FIG415_MOTION}）。"
              "\n       兩圖將幾乎無法分辨，讀者易視為重複排版。"
              "\n       建議改換其一。")

    pp = Path(args.temporal) / f"pose_{args.motion}.npz"
    if not pp.exists():
        sys.exit(f"[錯誤] 找不到 {pp}，請先執行 run_romp_batch.py")
    poses = np.load(pp)["poses"]

    asset = load_asset(work, tex)
    bind = load_bind(args.bind)
    J_rest, parents, _, _ = build_smplx(args.model_dir, bind["betas"])
    full = motion_to_full_pose(poses, bind["full_pose"])

    # 避開頭尾各 10%——該處多為預備與收勢，姿態變化小
    lo, hi = int(0.1 * len(full)), int(0.9 * len(full))
    idx = np.linspace(lo, hi - 1, args.cols).astype(int)
    print(f"[INFO] 動作 {args.motion}（{len(full)} 影格），"
          f"取樣 {list(int(i) for i in idx)}")

    driven = [drive(asset, bind, J_rest, parents, full[int(i)]) for i in idx]

    # 共用取景：各格大小恆定，否則讀者會將取景變動誤讀為深度方向的位移
    allv = np.concatenate(driven, axis=0)
    cx = (allv[:, 0].max() + allv[:, 0].min()) / 2
    cy = (allv[:, 1].max() + allv[:, 1].min()) / 2
    half = max(allv[:, 0].ptp(), allv[:, 1].ptp()) * 0.55

    tiles = []
    for v, i in zip(driven, idx):
        im = cv2.cvtColor(render(v, asset, (cx, cy, half),
                                 flip_y=not args.no_flip_y),
                          cv2.COLOR_RGB2BGR)
        b = np.full((32, im.shape[1], 3), 255, np.uint8)
        cv2.putText(b, f"frame {int(i)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([im, b]))

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    p = od / "fig_4-13_final_avatar.png"
    cv2.imwrite(str(p), np.hstack(tiles))
    print(f"\n[完成] {p}（{args.cols} 格，共用取景）")
    print(f"\n圖說應載明：資產 {work.name}、動作段 {args.motion}、"
          f"各格共用取景。\n圖 4-15 所用者為 {FIG415_WORK} / {FIG415_MOTION}，"
          "兩圖的資產與動作不同，宜於圖說中各自標明以免讀者誤認重複。")


if __name__ == "__main__":
    main()
