r"""
run_laplacian_sweep.py —— 軸 D：拉普拉斯平滑權重的取捨曲線

對應論文 4.2.3 節新增之軸 D。取代原 4.1 節的兩值比較（權重 40 vs 100）。

為何不能併入現有的三軸掃描
==========================
run_factor_sweep.py 的早停機制在第一階段法向量圖寫出後即終止行程，
**網格雕刻根本不會執行**——而拉普拉斯平滑權重正是作用於雕刻階段。
因此軸 D 的每個樣本都必須完整生成（約 10.9 分鐘，見論文表 4-15），
無法沿用早停路徑。5 個權重 × 3 組提示 = 15 次，約 2.7 小時。

為何需要第二個指標
==================
**粗糙度對平滑權重是單調的**——權重越高必然越平滑。若只量粗糙度，
「最佳值」的答案永遠是「用最大值」，該問題本身即不成立。

要使「最佳值」有意義，必須有一個會**懲罰過度平滑**的指標。本腳本採用
「著衣網格相對於 SMPL-X 裸體先驗的平均偏移量」：過度平滑會把衣服往
身體拉回、損失服裝體積，此量隨之下降。

於是取捨曲線為：
    粗糙度  ↓（好）   vs   服裝體積偏移  ↓（壞）
最佳點落在粗糙度改善趨緩、而體積尚未明顯流失之處。

此設計亦能回答「不同服裝類別是否有不同最佳值」：寬鬆的連身裙其體積
損失速度應快於貼身上衣，若成立則各類別的最佳權重確實不同。

尺度處理
========
Chupa 輸出為系統內部世界座標（身高約 0.85），SMPL-X 先驗為公尺（約 1.65）。
兩者尺度不同，故偏移量一律**以身高正規化**後再比較——與論文 4.5.5 節
厚度分布表的處理方式一致。

用法
====
  # 1) 確認 Chupa 設定中拉普拉斯權重的鍵名（必做，不可略過）
  python run_laplacian_sweep.py --stage probe

  # 2) 生成（約 2.7 小時，建議掛夜間）
  python run_laplacian_sweep.py --stage generate --out lap_sweep

  # 3) 量測與報告
  python run_laplacian_sweep.py --stage measure --out lap_sweep
  python run_laplacian_sweep.py --stage report  --out lap_sweep

環境：chupa
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
CHUPA_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
CHUPA_SCRIPT = "scripts/chupa.py"
CHUPA_CONFIG = "configs/text.yaml"
SUBJECT = "rp_emily_apose_001"
SUBJECT_PKL = "data/renderpeople/smplx/rp_emily_apose_001.pkl"

# 已由 --stage probe 確認：configs/text.yaml 第 70 行 `weight_laplacian: 40.0`，
# 巢狀於 nds.loss 之下。（初版誤猜為 nds.loss_weights.laplacian。）
CHUPA_ARG_LAPLACIAN = "nds.loss.weight_laplacian"

CHUPA_ARG_PROMPT = "chupa.prompt"
CHUPA_ARG_SEED = "chupa.seed"
CHUPA_ARG_RESAMPLE = "chupa.n_resample_from_text"
CHUPA_ARG_RESAMPLE_REFINE = "chupa.n_resample"

# 以浮點數表示，與設定檔中的 `40.0` 型別一致——Hydra 對結構化設定的
# 型別檢查可能拒絕以整數覆寫浮點欄位。
LAPLACIAN_VALUES = [20.0, 40.0, 60.0, 80.0, 100.0]
PROMPTS = [
    ("tshirt_shorts", "a person wearing a t-shirt and shorts"),
    ("long_dress",    "a woman wearing a long dress"),
    ("sleeveless",    "a person wearing a sleeveless top and long pants"),
]
SEED = 42
TIMEOUT = 3600


# ============================================================
# stage: probe —— 找出拉普拉斯權重的鍵名
# ============================================================

def yaml_key_path(path, lineno):
    r"""由 YAML 的縮排推出某一行的完整鍵路徑。

    Hydra 的命令列覆寫需要含所有層級的點分路徑（如 `nds.weight_laplacian`），
    而搜尋只能給出鍵名本身。此函式自該行往回掃，逐次尋找縮排更淺的鍵，
    組出完整路徑——比人工翻閱設定檔可靠，也避免再次猜錯。
    """
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return None
    if not (1 <= lineno <= len(lines)):
        return None

    def indent_of(s):
        return len(s) - len(s.lstrip(" "))

    target = lines[lineno - 1]
    key = target.strip().split(":")[0].strip()
    if not key:
        return None

    parts = [key]
    cur = indent_of(target)
    for j in range(lineno - 2, -1, -1):
        s = lines[j]
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        ind = indent_of(s)
        if ind < cur and ":" in s:
            parent = s.strip().split(":")[0].strip()
            if parent and not parent.startswith("-"):
                parts.append(parent)
                cur = ind
                if ind == 0:
                    break
    return ".".join(reversed(parts))


def stage_probe(args):
    r"""在 Chupa 設定檔中搜尋拉普拉斯權重的鍵名。

    此步驟不可略過：鍵名若填錯，Hydra 多半會靜默忽略該覆寫（或報錯），
    使五個條件實際上跑出相同的設定，而表面上看起來一切正常。
    """
    root = Path(CHUPA_PROJECT_DIR)
    pats = re.compile(r"(laplac|smooth|regular)", re.I)

    print(f"搜尋 {root}/configs 與 {root}/lib 之設定與原始碼\n")
    hits = []
    for sub in ("configs", "lib", "scripts", "src"):
        d = root / sub
        if not d.exists():
            continue
        for f in list(d.glob("**/*.yaml")) + list(d.glob("**/*.py")):
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if pats.search(line):
                        hits.append((f.relative_to(root), i, line.strip()))
            except OSError:
                continue

    if not hits:
        sys.exit("[未找到] 未出現 laplacian / smooth / regular 相關字樣。\n"
                 "請手動檢視 configs/text.yaml 與網格雕刻的實作。")

    yaml_hits = [h for h in hits if str(h[0]).endswith(".yaml")]
    print(f"設定檔（{len(yaml_hits)} 處）——鍵名最可能在此：\n")
    for f, i, line in yaml_hits[:30]:
        path = yaml_key_path(root / f, i)
        print(f"  {f}:{i}")
        print(f"      {line}")
        if path:
            print(f"      → Hydra 覆寫路徑：{path}")

    print(f"\n原始碼（{len(hits) - len(yaml_hits)} 處，供對照參數如何被讀取）：\n")
    for f, i, line in [h for h in hits if not str(h[0]).endswith(".yaml")][:20]:
        print(f"  {f}:{i}\n      {line}")

    print(f"\n目前腳本中的設定：CHUPA_ARG_LAPLACIAN = \"{CHUPA_ARG_LAPLACIAN}\"")
    print("\n請依上列輸出確認完整的鍵路徑（含所有層級，如 nds.loss_weights.laplacian），")
    print("填入腳本後再執行 --stage generate。")
    print("\n驗證方式：先以兩個差距極大的權重各跑一次，比較輸出網格的粗糙度；")
    print("若兩者完全相同，代表覆寫未生效，鍵名仍然錯誤。")


# ============================================================
# stage: generate
# ============================================================

def build_cmd(lap, prompt, seed):
    return [CHUPA_PYTHON, CHUPA_SCRIPT, CHUPA_CONFIG,
            f"dataset.subject={SUBJECT}",
            f"{CHUPA_ARG_PROMPT}={prompt}",
            f"{CHUPA_ARG_SEED}={seed}",
            f"{CHUPA_ARG_LAPLACIAN}={lap}",
            f"{CHUPA_ARG_RESAMPLE}=0",
            f"{CHUPA_ARG_RESAMPLE_REFINE}=0"]


def stage_generate(args):
    out = Path(args.out) / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    total = len(LAPLACIAN_VALUES) * len(PROMPTS)
    n = 0
    t_all = time.time()

    print(f"軸 D：{len(LAPLACIAN_VALUES)} 個權重 × {len(PROMPTS)} 組提示 = {total} 次完整生成")
    print(f"預估 {total * 10.9 / 60:.1f} 小時（每次約 10.9 分鐘）\n")

    for lap in LAPLACIAN_VALUES:
        for tag, prompt in PROMPTS:
            n += 1
            dst = out / f"lap{lap}_{tag}.obj"
            if dst.exists() and not args.force:
                print(f"[{n}/{total}] lap={lap} {tag} 已存在，略過")
                manifest.append({"lap": lap, "prompt": tag, "obj": str(dst)})
                continue

            print(f"[{n}/{total}] lap={lap} {tag} 生成中 ...", flush=True)
            t = time.time()
            r = subprocess.run(build_cmd(lap, prompt, SEED),
                               cwd=CHUPA_PROJECT_DIR, capture_output=True,
                               text=True, timeout=TIMEOUT)
            if r.returncode != 0:
                print(f"  ✗ 失敗：{r.stderr[-600:]}")
                continue

            cands = [p for p in (Path(CHUPA_PROJECT_DIR) / "results").rglob("mesh_refine.obj")
                     if p.stat().st_mtime >= t]
            if not cands:
                print("  ✗ 找不到輸出網格（確認雕刻階段是否執行）")
                continue

            import shutil
            shutil.copy(max(cands, key=lambda p: p.stat().st_mtime), dst)
            print(f"  完成（{(time.time() - t) / 60:.1f} 分鐘）")
            manifest.append({"lap": lap, "prompt": tag, "obj": str(dst)})

    (Path(args.out) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[完成] {len(manifest)} 筆，總計 {(time.time() - t_all) / 60:.1f} 分鐘")

    # 覆寫是否生效的自檢。鍵名錯誤時 Hydra 多半靜默忽略，五個條件會跑出
    # 相同設定而輸出看似正常——此為最難察覺的失敗模式，故主動檢查。
    _verify_override_took_effect(manifest)


def _verify_override_took_effect(manifest):
    lo, hi = LAPLACIAN_VALUES[0], LAPLACIAN_VALUES[-1]
    pairs = {}
    for m in manifest:
        if m["lap"] in (lo, hi):
            pairs.setdefault(m["prompt"], {})[m["lap"]] = m["obj"]

    same = []
    for tag, d in pairs.items():
        if lo not in d or hi not in d:
            continue
        a, b = Path(d[lo]), Path(d[hi])
        if a.exists() and b.exists() and a.stat().st_size == b.stat().st_size:
            if a.read_bytes() == b.read_bytes():
                same.append(tag)

    if same:
        print("\n" + "=" * 60)
        print(f"[警告] 權重 {lo} 與 {hi} 的輸出在下列提示上**逐位元相同**：")
        for t in same:
            print(f"    {t}")
        print("\n這代表拉普拉斯權重的覆寫未生效——Hydra 很可能忽略了鍵名。")
        print(f"目前使用的鍵名：{CHUPA_ARG_LAPLACIAN}")
        print("請重新執行 --stage probe，依「Hydra 覆寫路徑」修正後重跑。")
        print("=" * 60)
    elif pairs:
        print(f"\n[自檢通過] 權重 {lo} 與 {hi} 的輸出不同，覆寫已生效。")


# ============================================================
# stage: measure
# ============================================================

def normalized_roughness(mesh):
    from measure_roughness import roughness
    raw, elen, norm, n = roughness(mesh)
    return norm, n


def surface_area_ratio(mesh):
    r"""網格表面積除以其凸包表面積，作為皺褶保留度的指標。

    **為何需要此指標。** 初版僅以「相對於裸體先驗的平均偏移量」衡量過度平滑，
    實測顯示該量對平滑權重幾乎不變。但這並不代表皺褶未被抹平——平均偏移量
    反映的是服裝的**整體體積**，而皺褶是圍繞服裝表面的局部起伏，其正負位移
    在取平均後相互抵消，故該指標對皺褶不敏感。換言之，初版指標答錯了問題。

    表面積則直接對皺褶敏感：同樣的體積下，有皺褶的曲面其面積大於平滑曲面。
    除以凸包面積是為了正規化——不同體型與服裝的絕對尺寸不同，直接比較
    絕對面積會把尺寸差異誤計為皺褶差異。

    此比值下降即代表表面被抹平，是「過度平滑」的直接證據。
    """
    hull = mesh.convex_hull
    a_hull = float(hull.area)
    return float(mesh.area) / a_hull if a_hull > 0 else float("nan")


def mean_edge_length(mesh):
    e = np.asarray(mesh.edges_unique, np.int64)
    v = np.asarray(mesh.vertices, np.float64)
    return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())


def garment_offset(mesh, prior):
    r"""著衣網格相對於裸體先驗的平均偏移量，以身高正規化。

    兩者尺度不同（輸出為世界座標、先驗為公尺），故各自除以自身身高後
    再計算距離。此量隨過度平滑而下降——衣服被拉回貼近身體。
    """
    import trimesh

    def norm_unit(m):
        v = np.asarray(m.vertices, np.float64).copy()
        h = v[:, 1].max() - v[:, 1].min()
        v -= v.mean(axis=0)
        return v / max(h, 1e-9)

    mv = norm_unit(mesh)
    pv = norm_unit(prior)
    pm = trimesh.Trimesh(vertices=pv, faces=prior.faces, process=False)
    d = trimesh.proximity.signed_distance(pm, mv)
    return float(np.mean(np.abs(d))), float(np.percentile(np.abs(d), 90))


def stage_measure(args):
    from measure_girth import load_mesh, build_prior_mesh

    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    prior = build_prior_mesh(str(Path(CHUPA_PROJECT_DIR) / SUBJECT_PKL))

    recs = []
    for m in manifest:
        mesh = load_mesh(m["obj"])
        rough, nv = normalized_roughness(mesh)
        off_mean, off_p90 = garment_offset(mesh, prior)
        area_r = surface_area_ratio(mesh)
        elen = mean_edge_length(mesh)
        recs.append({**m, "roughness": rough, "n_verts": nv,
                     "offset_mean": off_mean, "offset_p90": off_p90,
                     "area_ratio": area_r, "mean_edge": elen})
        print(f"lap={m['lap']:>5} {m['prompt']:<16} 粗糙度 {rough:.5f}  "
              f"體積偏移 {off_mean:.5f}  面積比 {area_r:.4f}  ({nv:,} 頂點)")

    (out / "lap_results.json").write_text(
        json.dumps(recs, indent=2), encoding="utf-8")
    print(f"\n[完成] 已寫出 {out / 'lap_results.json'}")


# ============================================================
# stage: report
# ============================================================

def stage_report(args):
    out = Path(args.out)
    recs = json.loads((out / "lap_results.json").read_text(encoding="utf-8"))
    laps = sorted({r["lap"] for r in recs})
    prompts = [p for p, _ in PROMPTS]

    def get(lap, p):
        for r in recs:
            if r["lap"] == lap and r["prompt"] == p:
                return r
        return None

    L = []
    L.append("# 軸 D：拉普拉斯平滑權重的取捨\n")
    L.append("> 粗糙度對權重單調遞減，故單看粗糙度無法定義最佳值；"
             "須與服裝體積偏移併看。體積偏移下降代表衣服被平滑拉回貼近身體，"
             "即過度平滑。\n")

    L.append("\n## 正規化網格粗糙度（越低越平滑）\n")
    L.append("| 權重 | " + " | ".join(prompts) + " |")
    L.append("|---|" + "---:|" * len(prompts))
    for lap in laps:
        cells = [f"{get(lap,p)['roughness']:.5f}" if get(lap, p) else "—" for p in prompts]
        L.append(f"| {lap} | " + " | ".join(cells) + " |")

    L.append("\n## 服裝體積偏移（以身高正規化，下降代表體積流失）\n")
    L.append("| 權重 | " + " | ".join(prompts) + " |")
    L.append("|---|" + "---:|" * len(prompts))
    for lap in laps:
        cells = [f"{get(lap,p)['offset_mean']:.5f}" if get(lap, p) else "—" for p in prompts]
        L.append(f"| {lap} | " + " | ".join(cells) + " |")

    L.append("\n## 表面積比（網格面積 ÷ 凸包面積，下降代表皺褶被抹平）\n")
    L.append("> 體積偏移對皺褶不敏感——皺褶的正負位移在取平均後相互抵消。"
             "表面積則直接對皺褶敏感：同體積下有皺褶的曲面面積較大。\n")
    L.append("| 權重 | " + " | ".join(prompts) + " |")
    L.append("|---|" + "---:|" * len(prompts))
    for lap in laps:
        cells = [f"{get(lap,p)['area_ratio']:.4f}" if get(lap, p) else "—" for p in prompts]
        L.append(f"| {lap:g} | " + " | ".join(cells) + " |")

    L.append("\n## 相對於最低權重的變化率\n")
    L.append("> 「兩者之比」一欄已移除：體積變化接近零，相除所得數值不穩定"
             "（實測出現 56、−81、−22 等值），不具解讀意義。\n")
    L.append("| 提示 | 粗糙度改善 | 體積變化 | 表面積比變化 | 頂點數變化 |")
    L.append("|---|---:|---:|---:|---:|")
    for p in prompts:
        a, b = get(laps[0], p), get(laps[-1], p)
        if not (a and b):
            continue
        dr = (a["roughness"] - b["roughness"]) / a["roughness"] * 100
        do = (b["offset_mean"] - a["offset_mean"]) / a["offset_mean"] * 100
        da = (b["area_ratio"] - a["area_ratio"]) / a["area_ratio"] * 100
        dn = (b["n_verts"] - a["n_verts"]) / a["n_verts"] * 100
        L.append(f"| {p} | −{dr:.1f}% | {do:+.1f}% | {da:+.1f}% | {dn:+.1f}% |")

    # 密度混淆的量級估計
    L.append("\n## 密度混淆的估計\n")
    L.append("> 以平均邊長正規化**並未完全消除密度依賴**。對平滑曲面而言，"
             "拉普拉斯座標約與頂點間距 $h$ 的平方成正比、邊長與 $h$ 成正比，"
             "相除後仍殘留一次 $h$。頂點數隨權重上升，故部分的粗糙度下降"
             "來自密度而非平滑。\n")
    L.append("| 提示 | 頂點數變化 | 密度可解釋之降幅（估計） | 觀測降幅 | 歸因於平滑者 |")
    L.append("|---|---:|---:|---:|---:|")
    for p in prompts:
        a, b = get(laps[0], p), get(laps[-1], p)
        if not (a and b):
            continue
        dr = (a["roughness"] - b["roughness"]) / a["roughness"] * 100
        # h ∝ 1/sqrt(N)；殘留一次 h，故密度可解釋的降幅約為 1 - sqrt(Na/Nb)
        dens = (1 - np.sqrt(a["n_verts"] / b["n_verts"])) * 100
        L.append(f"| {p} | {(b['n_verts']-a['n_verts'])/a['n_verts']*100:+.1f}% | "
                 f"約 {dens:.1f}% | {dr:.1f}% | 約 {dr - dens:.1f}% |")

    L.append("\n## 判讀指引\n")
    L.append("- **若三組提示的曲線形狀一致**，則單一權重即足以涵蓋各類服裝，"
             "應照實報告此否定結果，不宜強行區分類別差異。")
    L.append("- **若表面積比隨權重顯著下降**，即代表皺褶確實被抹平，"
             "取捨存在，最佳權重應取於面積比開始明顯下降之前。")
    L.append("- **若表面積比亦持平**，則於所測範圍內不存在可量測的取捨，"
             "採用預設權重 40 的理由不能再是「避免抹平皺褶」，"
             "而應改述為與生成模型預設值保持一致。")
    L.append("- 密度混淆的估計僅為量級參考，其推導假設曲面為平滑主導；"
             "實際網格兼有平滑曲率與高頻雜訊，兩者的密度依賴方向相反，"
             "故該估計應視為上界而非精確修正。")

    p = out / "lap_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[完成] 已寫出 {p}")


def main():
    ap = argparse.ArgumentParser(description="軸 D：拉普拉斯平滑權重掃描")
    ap.add_argument("--stage", required=True,
                    choices=["probe", "generate", "measure", "report"])
    ap.add_argument("--out", default="lap_sweep")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    {"probe": stage_probe, "generate": stage_generate,
     "measure": stage_measure, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    main()