r"""
run_pipeline.py

把整條「文字 prompt + 動作影片 → 帶材質的可動畫化身」串起來。

  文字 prompt ─┬→ chupa                     (chupa env)  → mesh_refine.obj
               ├→ uv_texture_bake render    (chupa env)
               ├→ uv_texture_bake semantic  (chupa env)
               ├→ uv_texture_bake generate  (chupa_tex env) ← 跨環境
               ├→ drive_avatar weights      (chupa env)
  影片 → ROMP ─┴→ drive_avatar export       (chupa env)  → avatar.glb

設計上的三個重點：

  * 跨環境呼叫。generate 那步需要新版 diffusers，跟 chupa 的舊 torch 不相容，
    所以用 `conda run -n <env>` 從外部驅動，而不是硬擠進同一個環境。

  * 每個階段都可以單獨跳過（--skip）與單獨重跑（--only）。這條 pipeline 的
    後段（貼圖生成）要跑很久，調前段參數時不該被迫整條重跑。

  * chupa 的輸出路徑帶時間戳、不可預測，所以用 glob 找最新的那個。
    找到什麼會印出來讓你確認，不會默默用錯檔案。

────────────────────────────────────────────────────────────────
用法
────────────────────────────────────────────────────────────────

  conda activate chupa          # 這支腳本本身在 chupa 環境跑
  cd ~/projects/chupa

  # 完整跑一次
  python run_pipeline.py \
      --prompt "a person wearing a red jacket and black pants" \
      --subject rp_ben_posed_001 \
      --motion ~/projects/romp_test/img4640_output \
      --out-dir avatar_out

  # 只重跑貼圖生成（幾何與權重沿用上次結果）
  python run_pipeline.py --only generate --out-dir avatar_out \
      --prompt "a person wearing a navy coat and khaki trousers"

  # 沿用之前手動跑出來的 tex_work，不用重跑 render
  python run_pipeline.py --only generate --work tex_work --out-dir avatar_out \
      --prompt "a person wearing a navy coat and khaki trousers"

  # 跳過 chupa（已經有 mesh 了），從貼圖開始
  python run_pipeline.py --skip chupa \
      --mesh results/.../mesh_refine.obj \
      --prompt "..." --subject rp_ben_posed_001 \
      --motion ~/projects/romp_test/img4640_output

  # 先不接動作，只出靜態綁定驗證
  python run_pipeline.py --prompt "..." --subject rp_ben_posed_001

────────────────────────────────────────────────────────────────
第一次跑之前
────────────────────────────────────────────────────────────────
建議加 --dry-run：貼圖生成會用平塗底色取代 SD，整條 pipeline 幾分鐘就跑完，
可以先確認所有路徑、跨環境呼叫、檔案交接都對，再花 GPU 時間跑真的生成。
"""

import argparse
import shlex
import subprocess
import time
from pathlib import Path

# 各階段一律用字面的 "python"，交給 conda run 在目標環境裡解析。
PY = "python"


# ============================================================
# 設定（依你的專案結構調整）
# ============================================================

GEOM_ENV = "chupa"          # 有 nvdiffrast / xatlas / smplx 的環境
GEN_ENV = "chupa_tex"       # 有新版 diffusers 的環境

CHUPA_SCRIPT = "scripts/chupa.py"
CHUPA_CONFIG = "configs/text.yaml"
CHUPA_RESULTS = "results/renderpeople"

SMPLX_MODEL_DIR = "src/smpl_related/models"
BIND_PKL_TMPL = "data/renderpeople/smplx/{subject}.pkl"
# chupa 產生的、對齊到輸出網格且維持 SMPL-X 拓樸的 obj。
# 命名慣例依你的 pipeline 而定，這裡沿用 compare_skinning_methods.py 的樣式。
SMPLX_OBJ_TMPL = "debug_source_smplx_{subject}_v2.obj"

STAGES = ["chupa", "render", "semantic", "generate", "weights", "drive"]


# ============================================================
# 工具
# ============================================================

class Step:
    """把每個階段的計時、標題、失敗訊息統一處理掉。"""

    def __init__(self, name, index, total):
        self.name, self.index, self.total = name, index, total

    def __enter__(self):
        self.t0 = time.time()
        print(f"\n{'=' * 60}")
        print(f"[{self.index}/{self.total}] {self.name}")
        print(f"{'=' * 60}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc_type is None:
            print(f"→ {self.name} 完成 ({dt:.1f}s)")
        else:
            print(f"→ {self.name} 失敗 ({dt:.1f}s)")
        return False


def run(cmd, env_name=None):
    """執行指令。給了 env_name 就透過 conda run 在該環境裡跑。

    用 conda run 而不是要求使用者手動切環境，是因為這條 pipeline 橫跨兩個
    互不相容的環境，手動切換既麻煩又容易在中途跑錯環境。

    注意 cmd 裡的直譯器一定要是字面的 "python"，不能用 sys.executable ——
    後者是「當前」環境的絕對路徑，conda run 會忠實地執行它，結果就是
    在目標環境的變數下跑了來源環境的直譯器，跨環境完全失效而且不會報錯。
    """
    if env_name:
        cmd = ["conda", "run", "-n", env_name, "--no-capture-output"] + cmd
    print("$ " + " ".join(shlex.quote(str(c)) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise RuntimeError(
            f"指令失敗（return code {r.returncode}）")


def need(path, what, hint):
    """檢查前置檔案存在，不存在就給出可行動的錯誤訊息。

    這條 pipeline 的階段之間靠檔案交接，缺檔時原生的 FileNotFoundError
    只會說「找不到 meta.json」，看不出是哪個階段沒跑或路徑指錯。
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到 {what}：{path}\n  → {hint}")


def find_latest_mesh(subject, prompt, results_root=CHUPA_RESULTS):
    """在 chupa 的輸出目錄裡找最新的 mesh_refine.obj。

    chupa 的輸出路徑是 results/<dataset>/<timestamp>/<subject>/<prompt_slug>/，
    時間戳不可預測，所以只能 glob 之後取最新。找到什麼一定印出來，
    避免默默拿到上一次實驗的結果。
    """
    root = Path(results_root)
    if not root.exists():
        raise FileNotFoundError(f"找不到 chupa 輸出目錄 {root}")

    cands = sorted(root.glob(f"*/{subject}/*/mesh_refine.obj"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(
            f"在 {root} 底下找不到 {subject} 的 mesh_refine.obj。\n"
            f"chupa 是不是還沒跑完，或 --subject 打錯了？")

    # prompt 通常會被轉成路徑片段，能對上就優先用，對不上退回最新的
    slug = "".join(c if c.isalnum() else "_" for c in (prompt or "").lower())[:40]
    if slug:
        for c in cands:
            if slug[:20] and slug[:20] in c.parent.name.lower():
                print(f"[INFO] 依 prompt 比對到: {c}")
                return c
    print(f"[INFO] 用最新的輸出: {cands[0]}")
    if len(cands) > 1:
        print(f"       (同一個 subject 底下還有 {len(cands) - 1} 個較舊的結果)")
    return cands[0]


# ============================================================
# 各階段
# ============================================================

def stage_chupa(args, ctx):
    # n_resample_from_text / n_resample 必須是 0。
    # 這兩處 resample 是以「裸體先驗」為條件的，開著會把文字生成的服裝
    # 洗回長袖長褲 —— 也就是文字換裝會整個失效，而且不會報錯，
    # 只會安靜地得到一件跟 prompt 無關的衣服。
    cmd = [PY, CHUPA_SCRIPT, CHUPA_CONFIG,
           f"chupa.prompt={args.prompt}",
           f"dataset.subject={args.subject}",
           "chupa.n_resample_from_text=0",
           "chupa.n_resample=0"]
    run(cmd, GEOM_ENV)
    ctx["mesh"] = find_latest_mesh(args.subject, args.prompt)


def stage_render(args, ctx):
    need(ctx["mesh"], "chupa 網格", "先跑 chupa 階段，或用 --mesh 指定")
    run([PY, "uv_texture_bake.py", "--stage", "render",
         "--mesh", ctx["mesh"], "--work", ctx["work"]], GEOM_ENV)


def stage_semantic(args, ctx):
    need(Path(ctx["work"]) / "geometry.npz", "render 階段的輸出",
         f"先跑 render 階段，或用 --work 指向已有的資料夾")
    need(ctx["smplx_obj"], "對齊的 SMPL-X obj", "用 --smplx-obj 指定正確路徑")
    run([PY, "uv_texture_bake.py", "--stage", "semantic",
         "--work", ctx["work"], "--smplx-obj", ctx["smplx_obj"],
         "--smplx-model-dir", args.smplx_model_dir,
         "--prompt", args.prompt], GEOM_ENV)
    print(f"\n[提示] 打開 {ctx['work']}/label_preview.png 確認上衣/褲子邊界。")
    print(f"       不對就調 uv_texture_bake.py 裡的 HEM_DROP 重跑 "
          f"--only semantic（幾秒鐘）。")


def stage_generate(args, ctx):
    need(Path(ctx["work"]) / "meta.json", "render 階段的輸出",
         f"先跑 render 階段，或用 --work 指向已有的 tex_work 資料夾")
    cmd = [PY, "uv_texture_bake.py", "--stage", "generate",
           "--work", ctx["work"], "--prompt", args.prompt,
           "--out", ctx["tex_out"]]
    if args.dry_run:
        cmd.append("--dry-run")
    run(cmd, GEN_ENV)      # ← 唯一需要新環境的階段


def stage_weights(args, ctx):
    need(Path(ctx["work"]) / "geometry.npz", "render 階段的輸出",
         f"先跑 render 階段，或用 --work 指向已有的資料夾")
    need(ctx["bind_pkl"], "bind pose 參數檔", "用 --bind-pkl 指定正確路徑")
    run([PY, "drive_avatar.py", "--stage", "weights",
         "--work", ctx["work"], "--smplx-obj", ctx["smplx_obj"],
         "--bind-pkl", ctx["bind_pkl"],
         "--smplx-model-dir", args.smplx_model_dir], GEOM_ENV)


def stage_drive(args, ctx):
    tex = Path(ctx["tex_out"]) / "chupa_textured_texture.png"
    need(tex, "貼圖", "先跑 generate 階段")
    need(Path(ctx["work"]) / "skin_weights.npz", "蒙皮權重", "先跑 weights 階段")

    cmd = [PY, "drive_avatar.py", "--stage", "export",
           "--work", ctx["work"], "--texture", tex,
           "--bind-pkl", ctx["bind_pkl"],
           "--smplx-model-dir", args.smplx_model_dir,
           "--out", ctx["glb"]]
    if args.motion:
        cmd += ["--motion", args.motion, "--fps", args.fps]
        if args.use_translation:
            cmd.append("--use-translation")
    else:
        print("[INFO] 沒給 --motion，只輸出靜態綁定")
    run(cmd, GEOM_ENV)


HANDLERS = {"chupa": stage_chupa, "render": stage_render,
            "semantic": stage_semantic, "generate": stage_generate,
            "weights": stage_weights, "drive": stage_drive}


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="文字 prompt + 動作影片 → 帶材質的可動畫化身")
    p.add_argument("--prompt", default="a person wearing a red jacket and black pants")
    p.add_argument("--subject", default="rp_ben_posed_001")
    p.add_argument("--motion", default=None,
                   help="ROMP 輸出資料夾。不給就只出靜態綁定")
    p.add_argument("--fps", default="30")
    p.add_argument("--use-translation", action="store_true",
                   help="套用 ROMP 位移讓角色在畫面裡移動。預設關閉 —— "
                        "位置由綁定資產固定，虛擬試衣不需要人物走動")
    p.add_argument("--out-dir", default="avatar_out",
                   help="所有中間檔與成品都放這裡")
    p.add_argument("--work", default=None,
                   help="階段之間的交換資料夾。預設 <out-dir>/tex_work，"
                        "想沿用手動跑出來的成果就指過去（例如 tex_work）")
    p.add_argument("--mesh", default=None,
                   help="直接指定 mesh_refine.obj（配合 --skip chupa）")
    p.add_argument("--smplx-obj", default=None,
                   help="對齊的 SMPL-X obj，不給就用命名慣例推導")
    p.add_argument("--bind-pkl", default=None,
                   help="bind pose 參數檔，不給就用命名慣例推導")
    p.add_argument("--smplx-model-dir", default=SMPLX_MODEL_DIR)
    p.add_argument("--only", nargs="+", choices=STAGES,
                   help="只跑這些階段")
    p.add_argument("--skip", nargs="+", choices=STAGES, default=[],
                   help="跳過這些階段")
    p.add_argument("--dry-run", action="store_true",
                   help="貼圖生成用平塗底色取代 SD，幾分鐘跑完整條流程驗證接線")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = {
        "work": args.work or str(out / "tex_work"),
        "tex_out": str(out / "texture"),
        "glb": str(out / ("avatar_animated.glb" if args.motion else "avatar.glb")),
        "mesh": args.mesh,
        "smplx_obj": args.smplx_obj or SMPLX_OBJ_TMPL.format(subject=args.subject),
        "bind_pkl": args.bind_pkl or BIND_PKL_TMPL.format(subject=args.subject),
    }

    todo = [s for s in (args.only or STAGES) if s not in args.skip]

    # 跳過 chupa 但沒給 mesh 時，自己去找 —— 常見情境是幾何已經跑好、
    # 只想重調貼圖，不該逼使用者手動貼那串帶時間戳的路徑。
    if "chupa" not in todo and "render" in todo and not ctx["mesh"]:
        ctx["mesh"] = str(find_latest_mesh(args.subject, args.prompt))

    for f in [ctx["smplx_obj"], ctx["bind_pkl"]]:
        if not Path(f).exists() and set(todo) & {"semantic", "weights"}:
            print(f"[WARN] 找不到 {f}，semantic / weights 階段會失敗。"
                  f"用 --smplx-obj / --bind-pkl 指定正確路徑。")

    print(f"prompt   : {args.prompt}")
    print(f"subject  : {args.subject}")
    print(f"motion   : {args.motion or '(無，只出靜態綁定)'}")
    print(f"交換資料夾: {Path(ctx['work']).resolve()}")
    print(f"輸出     : {out.resolve()}")
    print(f"要跑的階段: {' → '.join(todo)}")
    if args.dry_run:
        print("模式     : --dry-run（貼圖用平塗底色，不跑 SD）")

    t0 = time.time()
    for i, name in enumerate(todo, 1):
        with Step(name, i, len(todo)):
            HANDLERS[name](args, ctx)

    print(f"\n{'=' * 60}")
    print(f"全部完成，總耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    if Path(ctx["glb"]).exists():
        print(f"成品: {Path(ctx['glb']).resolve()}")
        print(f"      Blender: File → Import → glTF 2.0，匯入後按空白鍵播放")


if __name__ == "__main__":
    main()