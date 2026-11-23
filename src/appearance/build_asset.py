r"""
build_asset.py —— 由 Chupa 生成的網格建出完整的可驅動資產（一鍵五步）

背景
====
由 mesh_refine.obj 到可驅動資產需五個步驟，且**跨兩個 conda 環境**：

    render   (chupa)      UV 展開 + 多視角渲染
    semantic (chupa)      骨架語意標籤
    generate (chupa_tex)  多視角貼圖生成      ← 環境不同
    export   (chupa)      貼圖烘焙與輸出
    weights  (chupa)      蒙皮權重轉移

手動執行時，路徑須逐一填入、環境須逐段切換，極易出錯。本腳本自動完成
路徑尋找與直譯器選擇，**不需要手動填任何路徑**。

用法
====
  # 看有哪些生成結果可用（依時間新到舊排序）
  python build_asset.py --list

  # 用最新一次的生成結果建資產
  python build_asset.py --work demo_work/A1_red_dress_v2

  # 指定用哪一次生成（--match 為子字串比對，不需完整路徑）
  python build_asset.py --match floor-length --work demo_work/A1_red_dress_v2

  # 只重跑某幾步（例如貼圖不滿意，幾何不用重做）
  python build_asset.py --match floor-length --work demo_work/A1_red_dress_v2 \
      --steps generate export

提示詞會自動由生成目錄的名稱還原，故不需手動輸入，也不會與生成時不一致。

環境：任一（本腳本自身只用標準函式庫，各步驟會自行選用正確的直譯器）
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PY = {
    "chupa":     os.path.expanduser("~/miniconda3/envs/chupa/bin/python"),
    "chupa_tex": os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python"),
}

# 各步驟所需的環境。render/semantic/export 需 xatlas 與 nvdiffrast（僅存在於
# chupa）；generate 需新版擴散模型函式庫（僅存在於 chupa_tex）。兩者相依性
# 無法調和，故不可於單一環境跑完整條流程。
STEP_ENV = {
    "render":   "chupa",
    "semantic": "chupa",
    "generate": "chupa_tex",
    "export":   "chupa",
    "weights":  "chupa",
}
ALL_STEPS = ["render", "semantic", "generate", "export", "weights"]

RESULTS = ROOT / "results" / "renderpeople"
BIND_DIR = ROOT / "data" / "renderpeople" / "smplx"


def find_meshes():
    """回傳所有已完成的生成結果，依修改時間新到舊排序。"""
    out = []
    if not RESULTS.is_dir():
        return out
    for m in RESULTS.glob("*/*/*/mesh_refine.obj"):
        # 路徑形如 results/renderpeople/<時戳>/<subject>/<提示_種子>/mesh_refine.obj
        subject = m.parent.parent.name
        tag = m.parent.name
        out.append({
            "mesh": m,
            "subject": subject,
            "tag": tag,
            "prompt": prompt_from_tag(tag),
            "mtime": m.stat().st_mtime,
        })
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def prompt_from_tag(tag):
    """由目錄名還原提示詞：a_woman_wearing_a_red_dress_0042 → 原句。

    自動還原可確保貼圖階段所用的提示詞與生成幾何時完全一致——兩者不一致
    會使貼圖的服裝語意與幾何對不上，且此種錯誤不會報錯，只會安靜地產出
    品質較差的結果。
    """
    s = re.sub(r"_\d+$", "", tag)          # 去掉結尾的種子
    return s.replace("_", " ")


def find_smplx_obj(subject):
    """尋找與該 subject 對齊的 SMPL-X obj。"""
    cands = sorted(ROOT.glob(f"debug_source_smplx_{subject}*.obj"))
    if not cands:
        return None
    # 若有 _v2 之類的修訂版，取名稱最長者（通常為較新的修訂）
    return max(cands, key=lambda p: len(p.name))


def find_bind(subject):
    p = BIND_DIR / f"{subject}.pkl"
    return p if p.exists() else None


def run(env, args, dry):
    cmd = [PY[env]] + args
    print(f"\n$ ({env}) " + " ".join(str(a) for a in cmd[1:]))
    if dry:
        return True
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[中止] 上一步失敗（return code {r.returncode}）。"
              f"後續步驟依賴其輸出，故不繼續。")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description="由生成網格建出可驅動資產")
    ap.add_argument("--work", help="資產輸出目錄，例如 demo_work/A1_red_dress_v2")
    ap.add_argument("--match", help="以子字串挑選生成結果；省略則用最新的一次")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS)
    ap.add_argument("--list", action="store_true", help="列出可用的生成結果")
    ap.add_argument("--dry-run", action="store_true", help="只印出指令，不執行")
    args = ap.parse_args()

    meshes = find_meshes()
    if not meshes:
        sys.exit(f"[錯誤] 於 {RESULTS} 下找不到任何 mesh_refine.obj")

    if args.list:
        print(f"可用的生成結果（共 {len(meshes)} 筆，新到舊）：\n")
        for i, m in enumerate(meshes[:25]):
            print(f"  [{i}] {m['subject']}　{m['tag']}")
            print(f"      提示：{m['prompt']}")
        return

    if not args.work:
        sys.exit("[錯誤] 請以 --work 指定資產輸出目錄")

    if args.match:
        sel = [m for m in meshes if args.match in m["tag"]]
        if not sel:
            sys.exit(f"[錯誤] 沒有 tag 含「{args.match}」的生成結果；"
                     f"可執行 python build_asset.py --list 查看")
        if len({m["tag"] for m in sel}) > 1:
            print(f"[警告] 「{args.match}」符合多筆，採用最新的一筆：")
            for m in sel[:5]:
                print(f"    {m['tag']}")
        chosen = sel[0]
    else:
        chosen = meshes[0]
        print("[INFO] 未指定 --match，採用最新的一次生成")

    work = Path(args.work)
    prompt = chosen["prompt"]
    subject = chosen["subject"]

    smplx_obj = find_smplx_obj(subject)
    bind = find_bind(subject)

    print("=" * 68)
    print(f"  生成結果 : {chosen['tag']}")
    print(f"  網格     : {chosen['mesh'].relative_to(ROOT)}")
    print(f"  提示詞   : {prompt}")
    print(f"  對象     : {subject}")
    print(f"  SMPL-X   : {smplx_obj.name if smplx_obj else '找不到'}")
    print(f"  綁定參數 : {bind.name if bind else '找不到'}")
    print(f"  輸出至   : {work}")
    print(f"  執行步驟 : {' → '.join(args.steps)}")
    print("=" * 68)

    if "semantic" in args.steps and not smplx_obj:
        sys.exit(f"[錯誤] 找不到 debug_source_smplx_{subject}*.obj，"
                 f"semantic 步驟需要它")
    if "weights" in args.steps and not bind:
        sys.exit(f"[錯誤] 找不到 {BIND_DIR}/{subject}.pkl，weights 步驟需要它")

    # 只有「已含實際內容」的資產才需確認——空目錄多為前次失敗所遺留，
    # 對其提問只會製造無意義的中斷。
    if (work / "geometry.npz").exists() and "render" in args.steps \
            and not args.dry_run:
        # render 會重建 geometry.npz，覆蓋既有資產。既有資產可能是論文
        # 已引用的結果，覆蓋後無法復原，故此處要求明確確認。
        ans = input(f"\n{work} 已存在，render 步驟會覆蓋其幾何。繼續？[y/N] ")
        if ans.strip().lower() != "y":
            sys.exit("已取消。若只要重跑貼圖，可用 --steps generate export")

    work.mkdir(parents=True, exist_ok=True)
    bake = "uv_texture_bake.py"

    plan = {
        "render":   [bake, "--stage", "render", "--mesh", chosen["mesh"],
                     "--work", work, "--prompt", prompt],
        "semantic": [bake, "--stage", "semantic", "--work", work,
                     "--smplx-obj", smplx_obj] if smplx_obj else None,
        "generate": [bake, "--stage", "generate", "--work", work,
                     "--prompt", prompt],
        "export":   [bake, "--stage", "export", "--work", work, "--out", work],
        "weights":  ["drive_avatar.py", "--stage", "weights", "--work", work,
                     "--smplx-obj", smplx_obj, "--bind", bind]
                    if (smplx_obj and bind) else None,
    }

    for step in args.steps:
        if plan[step] is None:
            sys.exit(f"[錯誤] {step} 步驟缺少必要檔案")
        if not run(STEP_ENV[step], plan[step], args.dry_run):
            sys.exit(1)

    if args.dry_run:
        print("\n[dry-run] 未實際執行。")
        return

    print(f"\n[完成] 資產已建於 {work}")
    missing = [f for f in ("geometry.npz", "skin_weights.npz", "texture.png")
               if not (work / f).exists()]
    if missing:
        print(f"  [警告] 仍缺少：{', '.join(missing)}")
    else:
        print("  geometry.npz、skin_weights.npz、texture.png 皆已產出，可用於圖 4-16。")
        print(f"\n接著把 make_demo_figures.py 的 OUTFITS 第一項改為 {work}，再執行：")
        print("  python make_demo_figures.py --fig 16 --out figures")


if __name__ == "__main__":
    main()
