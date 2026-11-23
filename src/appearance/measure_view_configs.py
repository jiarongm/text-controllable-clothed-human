r"""
measure_view_configs.py —— 快速與品質兩組視角配置的對照量測

補齊論文表 4-15 的兩個待補格：品質組態（50 視角）的貼圖耗時，
以及**快速組態（12 視角）的 UV 覆蓋率**。

為何需要這支腳本
================
論文原本混用兩組設定而未言明：3.8.2 節寫 50 視角，4.5.2 至 4.5.4 的消融
與 4.5.6 的成本量測卻用 12 視角，而 97.5% 的覆蓋率未標示所屬組態。
此矛盾會使讀者無從判斷「系統的正式設定究竟為何」。

本腳本將兩組態各量一次，使其成為一個明確的速度—品質取捨，而非設定不一致。

**其中快速組態的覆蓋率是關鍵數字**：
  * 若仍接近 97.5% —— 品質組態的額外 38 個視角並未帶來實質的覆蓋改善，
    3.8.2 節的預設配置本身即應重新檢討；
  * 若明顯較低（如低於 90%）—— 兩組態的取捨具體成立，快速組態僅宜用於
    消融與預覽，不宜作為最終資產。

兩種結果都是有意義的發現，皆須照實報告。

不需重新生成幾何
================
兩項量測皆沿用既有的著衣網格（`mesh_refine.obj`），僅重跑 UV 展開之後的
貼圖流程，成本為分鐘級。

**各組態使用獨立的工作目錄**：render 階段會重新執行 UV 展開並寫出
`geometry.npz`，若指向既有資產的工作目錄，將覆蓋其幾何與 UV，使之與已生成
的貼圖、蒙皮權重不再對應。本腳本一律於 `--out` 之下另建目錄，不觸碰既有資產。

跨環境執行
==========
本腳本**不可**於單一環境執行完畢——這正是 uv_texture_bake.py 拆分階段的前提：

  * `render`、`semantic` —— 需 xatlas 與 nvdiffrast，僅存在於 **chupa**
  * `generate`           —— 需新版擴散模型函式庫，僅存在於 **chupa_tex**

兩者的相依性無法調和。本腳本為各階段指定所屬環境的直譯器，故**於任一環境
啟動皆可**，其自身僅需標準函式庫與 numpy。

用法
====
  # 最簡：自動選用最新的 mesh_refine.obj
  python measure_view_configs.py --prompt "a navy coat and khaki pants"

  # 先看有哪些來源網格可選
  python measure_view_configs.py --list-meshes

  # 指定網格（兩組態務必使用同一個）
  python measure_view_configs.py --prompt "..." \
      --mesh results/renderpeople/20260726-195948/rp_emily_apose_001/\
a_woman_wearing_a_long_dress_0042/mesh_refine.obj

`--smplx-obj` 可省略，將沿用 uv_texture_bake.py 的預設值。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
UVBAKE = "uv_texture_bake.py"

# 各階段所屬的執行環境。此拆分正是 uv_texture_bake.py 的設計前提：
#   render / semantic —— 需 xatlas 與 nvdiffrast，僅存在於 chupa
#   generate          —— 需新版擴散模型函式庫，僅存在於 chupa_tex
# 兩者的相依性無法調和，故不可於單一環境跑完整條流程。
PY_CHUPA = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
PY_CHUPA_TEX = os.path.expanduser("~/miniconda3/envs/chupa_tex/bin/python")

STAGE_PYTHON = {
    "render":   PY_CHUPA,
    "semantic": PY_CHUPA,
    "generate": PY_CHUPA_TEX,
}

# 對應 uv_texture_bake.py 的 UVBAKE_FAST 開關
CONFIGS = {
    "fast":    {"env_fast": "1", "views": 12, "label": "快速組態"},
    "quality": {"env_fast": "",  "views": 50, "label": "品質組態"},
}


def run_stage(stage, work, fast, prompt=None, mesh=None, extra=None):
    r"""以該階段所屬環境的直譯器執行 uv_texture_bake 的一個階段。"""
    env = dict(os.environ)
    env["UVBAKE_FAST"] = fast
    py = STAGE_PYTHON[stage]
    if not Path(py).exists():
        print(f"  ✗ 找不到直譯器：{py}")
        print(f"    請確認環境名稱，或以 --py-chupa / --py-chupa-tex 指定")
        return None, 0.0, ""

    cmd = [py, UVBAKE, "--stage", stage, "--work", str(work)]
    if mesh:
        cmd += ["--mesh", str(mesh)]
    if prompt:
        cmd += ["--prompt", prompt]
    if extra:
        cmd += extra

    t = time.time()
    r = subprocess.run(cmd, cwd=CHUPA_PROJECT_DIR, env=env,
                       capture_output=True, text=True)
    dt = time.time() - t
    if r.returncode != 0:
        print(f"  ✗ {stage} 失敗：{r.stderr[-800:]}")
        return None, dt, r.stdout
    return True, dt, r.stdout


def _pct_in(line):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
    return float(m.group(1)) if m else None


def parse_coverage(stdout):
    r"""自 uv_texture_bake 的輸出解析**真實表面覆蓋率**。

    須嚴格指定所抓的行，因為輸出中有三個百分比，意義完全不同：

      1. render 階段：「UV atlas 實際佔用面積: X%」
         —— 此為 xatlas 打包後圖集佔整張貼圖的比例，**只取決於 UV 展開，
            與視角數無關**。初版解析誤抓此值，導致兩組態得到完全相同的
            數字（54.2%），並差點據以得出「增加視角無助於覆蓋」的錯誤結論。
      2. generate 階段：「覆蓋率（相對整張貼圖）: X%」
         —— 分母含打包空隙，會使健康的結果看似嚴重缺漏（論文 4.5.1 節）。
      3. generate 階段：「覆蓋率（相對 UV atlas 有效面積…）: X%」
         —— **此為唯一正確的表面覆蓋率**，即論文所報告的 97.5%。

    本函式只認第 3 種；找不到即回傳 None 並由呼叫端提示，不退而求其次。
    """
    real = raw = atlas = None
    for line in stdout.splitlines():
        if "有效面積" in line and "覆蓋率" in line:
            real = _pct_in(line)
        elif "覆蓋率" in line and "整張貼圖" in line:
            raw = _pct_in(line)
        elif "atlas 實際佔用面積" in line:
            atlas = _pct_in(line)
    return {"real": real, "raw": raw, "atlas": atlas}


def main():
    ap = argparse.ArgumentParser(description="兩組視角配置的對照量測")
    ap.add_argument("--mesh",
                    help="來源網格（mesh_refine.obj）。省略時自動選用最新的一個")
    ap.add_argument("--list-meshes", action="store_true",
                    help="列出可用的 mesh_refine.obj 後結束")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="view_configs")
    ap.add_argument("--smplx-obj", help="semantic 階段所需的 SMPL-X obj")
    ap.add_argument("--only", choices=["fast", "quality"])
    ap.add_argument("--py-chupa", default=PY_CHUPA)
    ap.add_argument("--py-chupa-tex", default=PY_CHUPA_TEX)
    args = ap.parse_args()

    STAGE_PYTHON["render"] = args.py_chupa
    STAGE_PYTHON["semantic"] = args.py_chupa
    STAGE_PYTHON["generate"] = args.py_chupa_tex

    # 自動尋找來源網格。兩組態必須使用**同一個**網格，否則所比較的
    # 就不只是視角數，覆蓋率的差異將無法歸因。
    root = Path(CHUPA_PROJECT_DIR)
    if args.list_meshes or not args.mesh:
        cands = sorted(root.glob("results/**/mesh_refine.obj"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            sys.exit("[錯誤] 找不到任何 mesh_refine.obj")
        if args.list_meshes:
            print(f"可用的來源網格（依修改時間排序，共 {len(cands)} 個）：\n")
            for p in cands[:20]:
                print(f"  {p.stat().st_size / 1e6:6.1f} MB  {p.relative_to(root)}")
            return
        args.mesh = str(cands[0].relative_to(root))
        print(f"[自動選用] --mesh {args.mesh}")
        print("  （如需指定其他網格，請以 --list-meshes 查看後手動給定）\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = {}

    todo = [args.only] if args.only else ["fast", "quality"]
    for name in todo:
        cfg = CONFIGS[name]
        print(f"\n=== {cfg['label']}（{cfg['views']} 視角）===")

        # **各組態使用獨立的工作目錄。**
        # render 階段會重新執行 UV 展開並寫出 geometry.npz；若指向既有資產的
        # 工作目錄，將覆蓋該資產的幾何與 UV，使其與已生成的貼圖、蒙皮權重
        # 不再對應。故此處一律另建目錄，不觸碰既有資產。
        work = out / f"work_{name}"
        work.mkdir(parents=True, exist_ok=True)

        print("  render ...")
        ok, t_render, so_r = run_stage("render", work, cfg["env_fast"],
                                       mesh=args.mesh)
        if not ok:
            continue

        print("  semantic ...")
        extra = ["--smplx-obj", args.smplx_obj] if args.smplx_obj else None
        ok, t_sem, _ = run_stage("semantic", work, cfg["env_fast"],
                                 prompt=args.prompt, extra=extra)
        if not ok:
            continue

        print("  generate ...")
        ok, t_gen, so_g = run_stage("generate", work, cfg["env_fast"],
                                    prompt=args.prompt)
        if not ok:
            continue

        # 覆蓋率只自 generate 的輸出解析——render 階段印的是 UV 圖集的
        # 佔用面積，與視角數無關，抓錯會使兩組態得到相同的數字。
        c = parse_coverage(so_g)
        c_render = parse_coverage(so_r)
        cov = c["real"]

        t_tex = t_render + t_sem + t_gen
        recs[name] = {"views": cfg["views"], "label": cfg["label"],
                      "t_render_s": t_render, "t_semantic_s": t_sem,
                      "t_generate_s": t_gen, "t_texture_s": t_tex,
                      "coverage_real": cov, "coverage_raw": c["raw"],
                      "atlas_area": c_render["atlas"], "work": str(work)}
        print(f"  渲染 {t_render:.1f} s、語意 {t_sem:.1f} s、生成 {t_gen:.1f} s、"
              f"貼圖流程合計 {t_tex:.1f} s")
        if cov is not None:
            print(f"  表面覆蓋率 {cov:.1f}%"
                  f"（相對整張貼圖 {c['raw']:.1f}%、"
                  f"圖集佔用面積 {c_render['atlas']:.1f}%）")
        else:
            print("  [警告] 未能解析表面覆蓋率——請自 generate 的輸出中"
                  "找「相對 UV atlas 有效面積」該行手動填入")

    (out / "view_configs.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")

    # 產出可直接貼入論文表 4-15 的列
    print("\n可貼入論文表 4-15：\n")
    print("| 組態 | 視角數 | 貼圖生成耗時 | UV 覆蓋率 | 單一資產總耗時 |")
    print("|---|---:|---:|---:|---:|")
    GEOM_MIN = 10.9    # 表 4-17 之幾何生成實測平均
    OTHER_MIN = 0.3    # UV 展開與語意標籤
    for name in ("fast", "quality"):
        r = recs.get(name)
        if not r:
            print(f"| {CONFIGS[name]['label']} | {CONFIGS[name]['views']} "
                  f"| — | — | — |")
            continue
        tex_min = r["t_texture_s"] / 60.0
        total = GEOM_MIN + OTHER_MIN + tex_min
        cov = (f"{r['coverage_real']:.1f}%"
               if r["coverage_real"] is not None else "手動填入")
        print(f"| {r['label']} | {r['views']} | {r['t_texture_s']:.0f} 秒 "
              f"| {cov} | {total:.1f} 分鐘 |")

    if len(recs) == 2:
        f, q = recs["fast"], recs["quality"]
        ratio = q["t_texture_s"] / max(f["t_texture_s"], 1e-9)
        gr = q["t_generate_s"] / max(f["t_generate_s"], 1e-9)
        print(f"\n貼圖流程耗時比：品質／快速 = {ratio:.2f}×"
              f"（視角數比 {q['views'] / f['views']:.2f}×）")
        print(f"其中 generate 階段之比為 {gr:.2f}×——"
              f"此階段才隨視角數縮放；render 含 UV 展開，其成本與視角數無關，"
              f"故合計的比值低於視角數之比。")

        a, b = f["coverage_real"], q["coverage_real"]
        if a is not None and b is not None:
            d = b - a
            print(f"\n表面覆蓋率：快速 {a:.1f}%、品質 {b:.1f}%（差 {d:+.1f} 個百分點）")
            if abs(d) < 2.0:
                print("\n[注意] 兩組態的表面覆蓋率相差不足 2 個百分點——"
                      "額外的 38 個視角並未帶來實質的覆蓋改善。"
                      "3.8.2 節之預設配置應重新檢討，論文須報告此一否定結果。"
                      "\n  惟報告前請先確認：兩組態的圖集佔用面積是否相同"
                      "（下方數值）？若相同，代表 UV 展開一致、比較有效；"
                      "若不同，則兩者的分母不同，覆蓋率不可直接比較。")
            else:
                print("\n[結論] 覆蓋率差異明確，兩組態的速度—品質取捨成立；"
                      "快速組態宜限於消融與預覽用途。")
            print(f"  圖集佔用面積：快速 {f['atlas_area']}%、"
                  f"品質 {q['atlas_area']}%")

    print(f"\n[完成] 已寫出 {out / 'view_configs.json'}")


if __name__ == "__main__":
    main()