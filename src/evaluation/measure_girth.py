r"""
measure_girth.py —— 軀幹橫截面量測：驗證「胖瘦」資訊是否通過生成流程

用途
====
論文 4.4.3 節已證實身高資訊完整保留（先驗比值 1.129 → 輸出比值 1.126）。
本腳本補上圍度面向，使 C3 由「身高保留」提升為「體型保留」。

為何不能用包圍盒寬度
====================
4.4.3 節已記錄此陷阱：A-pose 下包圍盒寬度實為雙臂展開的跨距，而臂展約等於
身高，故不論胖瘦，寬高比皆接近定值（實測 0.944 / 0.952 / 0.988）。真正的
圍度差異位於軀幹，被展開的手臂所掩蓋。

本腳本的做法
============
1. 於**身高比例**（非絕對高度）取胸／腰／臀三個高度切平面。
   用比例是必要的：若用絕對高度，矮的人量到胸線、高的人量到腰線，等於又把
   身高混進了圍度指標。
2. 以平面切割網格，取得橫截面多邊形。
3. **丟棄手臂的截面**：切平面在胸線高度會同時切到軀幹與雙臂，產生三個分離的
   封閉多邊形。本腳本保留最接近人體中軸者（即軀幹），捨棄其餘。
4. 對每個軀幹截面報告寬度、深度與截面積。
   截面積是三者中最穩健的圍度代理——寬度單獨看會漏掉前後向差異，而周長在
   著衣網格上易受布料皺褶的高頻起伏影響而被高估。

如何解讀輸出
============
著衣網格含布料厚度，其絕對值本就大於裸體先驗，**兩者的絕對值不可直接比較**。
可比較的是「體型之間的比值」——與 4.4.3 節身高的處理方式相同：

    先驗階段：胖者截面積 / 瘦者截面積 = R_prior
    輸出階段：胖者截面積 / 瘦者截面積 = R_output

若 R_output ≈ R_prior，即證實圍度資訊亦完整通過生成流程。

用法
====
  # 列出可用網格，協助確認路徑
  python measure_girth.py --stage list --root ~/projects/chupa

  # 量測單一網格
  python measure_girth.py --stage measure --mesh path/to/mesh.obj --label thin_prior

  # 完整對照：先驗 vs 輸出 × 多種體型
  # 欄位以 | 分隔；先驗寫 pkl:<路徑> 即自 betas 即時生成，
  # 不必在硬碟上尋找裸體網格。
  python measure_girth.py --stage compare \
      --pair "thin|pkl:bind_thin.pkl|work_thin/geometry.npz" \
      --pair "normal|pkl:bind_normal.pkl|work_normal/geometry.npz" \
      --pair "heavy|pkl:bind_heavy.pkl|work_heavy/geometry.npz" \
      --out girth_report.md

環境
====
於 chupa 環境執行（需 trimesh、numpy、shapely）。trimesh 的 section 需要
shapely 方能轉為平面多邊形；若未安裝：
  conda run -n chupa pip install shapely
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("[錯誤] 需要 trimesh。請於 chupa 環境執行：pip install trimesh")


def preflight(need_smplx=False):
    r"""啟動時檢查相依套件。

    trimesh 對缺少的套件採延遲報錯——直到真正呼叫 section() 時才在深層
    拋出 ModuleNotFoundError，訊息與實際原因相距甚遠。此處提前檢查，
    使錯誤訊息直接指向解法。

    注意本腳本須於 **chupa** 環境執行，而非 chupa_tex：
    後者為外觀生成而設，不含 scipy 與 smplx。
    """
    missing = []
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy")
    try:
        import shapely  # noqa: F401
    except ImportError:
        missing.append("shapely")
    if need_smplx:
        for mod in ("torch", "smplx"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)

    if missing:
        print("[錯誤] 缺少套件：" + "、".join(missing))
        print()
        print("本腳本須於 chupa 環境執行（chupa_tex 是給外觀生成用的，不含這些）：")
        print("    conda activate chupa")
        print()
        print("若 chupa 環境仍缺少，安裝指令：")
        print(f"    conda run -n chupa pip install {' '.join(missing)}")
        sys.exit(1)


# ============================================================
# 量測設定
# ============================================================

# 橫截面高度，以「自腳底起算的身高比例」表示。
# 數值採一般人體量測學的概略比例；重點不在絕對正確，而在三種體型之間一致。
SECTION_LEVELS = {
    "chest": 0.72,   # 胸線
    "waist": 0.62,   # 腰線（最細處附近）
    "hip":   0.52,   # 臀圍線
}

# 網格的「上」為哪一軸。SMPL-X 與 Chupa 輸出慣例為 Y-up。
UP_AXIS = 1

# 切平面若切不到（例如恰好落在網格孔洞上），於此範圍內微調重試
RETRY_OFFSETS = [0.0, 0.004, -0.004, 0.008, -0.008]


def load_mesh(spec):
    r"""載入網格。支援三種來源：

      1. 一般網格檔（.obj / .ply / .glb）
      2. `geometry.npz` —— 本專案的資產格式，drive_avatar.py 讀的就是這個。
         體型資產多以此格式存放，故不加此支援會在硬碟上找不到它們。
      3. `pkl:<路徑>` —— 自該 pkl 的 betas 即時生成 SMPL-X 裸體先驗。
         比在硬碟裡翻找先驗網格可靠：先驗本來就是由 betas 決定的，
         直接重建可確保與著衣輸出所用的體型參數完全一致。
    """
    s = str(spec)

    if s.startswith("pkl:"):
        return build_prior_mesh(s[4:])

    p = Path(os.path.expanduser(s))
    if p.suffix == ".npz":
        d = np.load(p)
        vk = next((k for k in ("verts", "vertices", "v") if k in d), None)
        fk = next((k for k in ("faces", "f") if k in d), None)
        if vk is None or fk is None:
            raise ValueError(f"{p} 缺少頂點或面片（現有鍵：{list(d.keys())}）")
        return trimesh.Trimesh(vertices=d[vk], faces=d[fk], process=False)

    m = trimesh.load(str(p), process=False, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"無法載入為單一網格：{p}")
    return m


def build_prior_mesh(pkl_path, model_dir="src/smpl_related/models"):
    """自 pkl 的 betas 重建 SMPL-X 裸體先驗（零姿態）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from drive_avatar import load_bind, build_smplx
    except ImportError as e:
        raise SystemExit(f"[錯誤] pkl: 模式需與 drive_avatar.py 同目錄：{e}")

    import smplx as smplx_lib

    bind = load_bind(os.path.expanduser(pkl_path))
    _, _, _, v_shaped = build_smplx(model_dir, bind["betas"])
    m = smplx_lib.create(model_dir, model_type="smplx", gender="neutral",
                         use_pca=True, num_pca_comps=6, num_betas=10,
                         num_expression_coeffs=10, batch_size=1)
    faces = m.faces.astype(np.int64)
    return trimesh.Trimesh(vertices=np.asarray(v_shaped, np.float64),
                           faces=faces, process=False)


def body_extent(mesh):
    """回傳 (腳底高度, 頭頂高度, 身高)。"""
    lo = float(mesh.vertices[:, UP_AXIS].min())
    hi = float(mesh.vertices[:, UP_AXIS].max())
    return lo, hi, hi - lo


def _plane_normal():
    n = [0.0, 0.0, 0.0]
    n[UP_AXIS] = 1.0
    return n


def torso_polygon(mesh, height):
    r"""於指定高度切一刀，回傳軀幹的截面多邊形（shapely Polygon）。

    切平面在胸線高度會同時切到軀幹與雙臂，得到多個分離的封閉多邊形，
    本函式須自其中選出軀幹。

    選取準則為**面積最大者**。曾考慮改以「最接近人體中軸者」為準，但
    to_planar() 會將截面轉入其自身的二維座標系，該座標系與世界座標的
    對應關係並不直接可得，故中軸準則實作上不可靠。面積準則不依賴座標
    系對應，且在 A-pose 下穩健——軀幹截面必大於單邊手臂截面。其失效
    情境見報告末段之限制說明。
    """
    origin = [0.0, 0.0, 0.0]
    origin[UP_AXIS] = height

    section = mesh.section(plane_origin=origin, plane_normal=_plane_normal())
    if section is None:
        return None

    # trimesh 於新版將 to_planar 更名為 to_2D，舊名於 2026/1/1 移除。
    # 兩者皆嘗試，以同時相容專案內的不同 trimesh 版本。
    to_2d = getattr(section, "to_2D", None) or getattr(section, "to_planar", None)
    if to_2d is None:
        return None
    try:
        planar, _ = to_2d()
    except Exception:
        return None

    polys = list(planar.polygons_full)
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]

    return max(polys, key=lambda p: p.area)


def measure_section(mesh, height):
    """回傳單一高度的量測結果 dict，失敗回傳 None。"""
    for off in RETRY_OFFSETS:
        poly = torso_polygon(mesh, height + off)
        if poly is not None and poly.area > 0:
            minx, miny, maxx, maxy = poly.bounds
            a, b = maxx - minx, maxy - miny
            return {
                "area": float(poly.area),
                "perimeter": float(poly.length),
                # 平面座標系的兩軸不保證對應世界的左右／前後，
                # 故以「較長者為寬、較短者為深」呈現，此對比值計算無影響
                "width": float(max(a, b)),
                "depth": float(min(a, b)),
                "offset_used": off,
            }
    return None


def measure_mesh(path, label=None):
    mesh = load_mesh(path)
    lo, hi, height = body_extent(mesh)
    out = {
        "label": label or Path(path).stem,
        "path": str(path),
        "n_verts": len(mesh.vertices),
        "height": height,
        "levels": {},
    }
    for name, frac in SECTION_LEVELS.items():
        h = lo + frac * height
        r = measure_section(mesh, h)
        out["levels"][name] = r
        if r is None:
            print(f"  [警告] {out['label']} 於 {name}（比例 {frac}）切不到截面")
    return out


# ============================================================
# stage: list
# ============================================================

def stage_list(args):
    r"""列出候選網格。

    預設會篩掉參數掃描留下的中間網格：`--stage generate` 於第一階段後即終止，
    其 `nds_dual/meshes/` 下的雙視角雕刻結果頂點數遠低於完整輸出（檔案約
    0.2 MB 對比數 MB），量測圍度會得到失真的截面。以檔案大小為門檻是最可靠
    的區辨方式——完整的著衣網格約七萬頂點，體積必然大得多。
    """
    root = Path(os.path.expanduser(args.root))
    if not root.exists():
        sys.exit(f"[錯誤] 找不到目錄：{root}")

    print(f"掃描 {root}")

    # 體型資產多以 geometry.npz 存放（drive_avatar.py 的格式），
    # 只掃 .obj 會完全找不到它們——先單獨列出。
    geos = sorted(root.glob("**/geometry.npz"))
    if geos:
        print(f"\n找到 {len(geos)} 個 geometry.npz（本專案的資產格式，"
              f"體型資產通常在此）：\n")
        for g in geos[:20]:
            print(f"  {g.stat().st_size / 1e6:7.1f} MB  {g.relative_to(root)}")
        if len(geos) > 20:
            print(f"  ...（其餘 {len(geos) - 20} 個省略）")
        print()

    # 一併列出可能的綁定參數檔，供 pkl: 模式使用
    pkls = sorted(root.glob("**/*.pkl"))
    pkls = [p for p in pkls if p.stat().st_size < 5e6]
    if pkls:
        print(f"找到 {len(pkls)} 個 .pkl（可用於 pkl:<路徑> 生成先驗）：\n")
        for p in pkls[:15]:
            print(f"           {p.relative_to(root)}")
        if len(pkls) > 15:
            print(f"  ...（其餘 {len(pkls) - 15} 個省略）")
        print()

    found = []
    for p in ("**/*.obj", "**/*.ply", "**/*.glb"):
        found.extend(root.glob(p))
    found = [f for f in found if f.stat().st_size > 0]
    total = len(found)

    kept = []
    for f in found:
        s = str(f)
        if any(x in s for x in args.exclude):
            continue
        if f.stat().st_size / 1e6 < args.min_mb:
            continue
        kept.append(f)

    print(f"共 {total} 個網格，篩選後剩 {len(kept)} 個"
          f"（排除樣式 {args.exclude}，大小門檻 {args.min_mb} MB）\n")
    if not kept:
        sys.exit("[錯誤] 篩選後沒有網格。請降低 --min-mb 或調整 --exclude。")

    # 先給目錄層級的摘要，避免大量同類檔案洗版
    groups = {}
    for f in kept:
        key = str(f.parent.relative_to(root))
        g = groups.setdefault(key, {"n": 0, "mb": 0.0})
        g["n"] += 1
        g["mb"] = max(g["mb"], f.stat().st_size / 1e6)

    print("目錄摘要（依最大檔案大小排序）：\n")
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["mb"])[:25]:
        print(f"  {g['mb']:7.1f} MB  ×{g['n']:<4d} {key}")

    print("\n最大的 25 個檔案：\n")
    kept.sort(key=lambda f: f.stat().st_size, reverse=True)
    for f in kept[:25]:
        print(f"  {f.stat().st_size / 1e6:7.1f} MB  {f.relative_to(root)}")

    print("\n你要找的是兩類檔案：")
    print("  1. SMPL-X 裸體先驗 —— 10,475 個頂點，三種體型各一")
    print("  2. 對應的 Chupa 著衣輸出 —— 約七萬頂點，三種體型各一")
    print("\n可先以 --stage measure 確認頂點數是否符合，再執行 --stage compare。")


# ============================================================
# stage: measure
# ============================================================

def stage_measure(args):
    r = measure_mesh(args.mesh, args.label)
    print(f"\n{r['label']}　頂點 {r['n_verts']:,}　身高（世界座標）{r['height']:.4f}\n")
    print(f"  {'部位':<8}{'寬度':>10}{'深度':>10}{'截面積':>12}{'周長':>10}")
    for name in SECTION_LEVELS:
        v = r["levels"][name]
        if v is None:
            print(f"  {name:<8}{'—':>10}{'—':>10}{'—':>12}{'—':>10}")
        else:
            print(f"  {name:<8}{v['width']:>10.4f}{v['depth']:>10.4f}"
                  f"{v['area']:>12.5f}{v['perimeter']:>10.4f}")


# ============================================================
# stage: compare
# ============================================================

def parse_pair(s):
    r"""解析 `標籤|先驗|輸出`。

    分隔符使用 `|` 而非 `:`，因為先驗欄位可寫成 `pkl:<路徑>`，其中已含冒號。
    """
    parts = s.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--pair 格式須為 標籤|先驗|輸出（以 | 分隔），收到：{s}")
    return parts[0], parts[1], parts[2]


def ratio_table(results, stage_key, metric):
    """以體型清單中的第一個為分母，計算各體型的比值。"""
    base = results[0][stage_key]["levels"]
    rows = []
    for r in results:
        lv = r[stage_key]["levels"]
        row = {"label": r["label"]}
        for name in SECTION_LEVELS:
            if lv[name] is None or base[name] is None or base[name][metric] == 0:
                row[name] = None
            else:
                row[name] = lv[name][metric] / base[name][metric]
        rows.append(row)
    return rows


def stage_compare(args):
    if len(args.pair) < 2:
        sys.exit("[錯誤] --stage compare 至少需要兩組體型（--pair）方能計算比值")

    results = []
    for label, prior_p, out_p in args.pair:
        print(f"量測 {label} ...")
        results.append({
            "label": label,
            "prior": measure_mesh(prior_p, f"{label}_prior"),
            "output": measure_mesh(out_p, f"{label}_output"),
        })

    lines = []
    add = lines.append

    add("# 軀幹橫截面量測結果\n")
    add("本表對應論文 4.4.3 節，補充身高之外的圍度面向。\n")
    add(f"截面高度（自腳底起算之身高比例）："
        + "、".join(f"{k} {v}" for k, v in SECTION_LEVELS.items()) + "\n")

    add("\n## 一、絕對量測值\n")
    add("> 著衣網格含布料厚度，其絕對值本就大於裸體先驗，"
        "**兩階段的絕對值不可直接比較**。本表僅供檢核量測是否合理。\n")
    add("| 體型 | 階段 | 身高 | 胸寬 | 胸深 | 胸截面積 | 腰截面積 | 臀截面積 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for stage_key, stage_name in (("prior", "SMPL-X 先驗"), ("output", "Chupa 輸出")):
            m = r[stage_key]
            lv = m["levels"]

            def g(name, key):
                return f"{lv[name][key]:.4f}" if lv[name] else "—"

            add(f"| {r['label']} | {stage_name} | {m['height']:.4f} | "
                f"{g('chest','width')} | {g('chest','depth')} | "
                f"{g('chest','area')} | {g('waist','area')} | {g('hip','area')} |")

    add("\n## 二、體型間比值（關鍵表）\n")
    add(f"> 以 **{results[0]['label']}** 為分母。若「輸出」列的比值接近「先驗」列，"
        "即代表圍度資訊完整通過生成流程——與 4.4.3 節身高的論證方式相同。\n")

    for metric, mname in (("area", "截面積"), ("width", "寬度")):
        add(f"\n**以{mname}計**\n")
        add("| 體型 | 階段 | 胸 | 腰 | 臀 |")
        add("|---|---|---:|---:|---:|")
        pr = ratio_table(results, "prior", metric)
        ou = ratio_table(results, "output", metric)
        for i, r in enumerate(results):
            for rows, sname in ((pr, "先驗"), (ou, "輸出")):
                row = rows[i]

                def f(name):
                    return f"{row[name]:.3f}" if row[name] is not None else "—"

                add(f"| {r['label']} | {sname} | {f('chest')} | {f('waist')} | {f('hip')} |")

        add("\n**保留度（兩階段比值之差，越接近 0 越好）**\n")
        add("| 體型 | 胸 | 腰 | 臀 |")
        add("|---|---:|---:|---:|")
        for i, r in enumerate(results):
            def d(name):
                a, b = pr[i][name], ou[i][name]
                if a is None or b is None:
                    return "—"
                return f"{abs(a - b):.4f}"
            add(f"| {r['label']} | {d('chest')} | {d('waist')} | {d('hip')} |")

    add("\n## 三、撰寫論文時須說明的限制\n")
    add("- 截面高度依身高比例取得，非依解剖特徵定位，故「腰線」未必落在最細處。"
        "此對**體型間比值**的影響有限（三種體型採同一比例），但不宜宣稱為標準人體量測。")
    add("- 著衣網格的截面含布料厚度與皺褶，其截面積略大於實際軀幹。"
        "此為兩階段絕對值不可比較的原因，亦是本表以比值呈現的理由。")
    add("- 軀幹截面以「面積最大者」自多個截面多邊形中選出。"
        "此準則在 A-pose 下穩健，惟若服裝極為寬鬆而使袖管截面大於軀幹，則可能誤選；"
        "建議以 --stage measure 逐一目視檢核數值是否合理。")

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[完成] 已寫出 {args.out}")
    else:
        print("\n" + text)


# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="軀幹橫截面量測：驗證圍度資訊是否通過生成流程")
    ap.add_argument("--stage", required=True,
                    choices=["list", "measure", "compare"])
    ap.add_argument("--root", default="~/projects/chupa",
                    help="--stage list 的掃描起點")
    ap.add_argument("--exclude", action="append",
                    default=["nds_dual/meshes"],
                    help="排除含此字串的路徑，可重複；預設排除參數掃描的中間網格")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="檔案大小門檻（MB），用以濾掉低頂點數的中間網格")
    ap.add_argument("--mesh", help="--stage measure 的目標網格")
    ap.add_argument("--label", help="--stage measure 的顯示名稱")
    ap.add_argument("--pair", action="append", type=parse_pair, default=[],
                    metavar="標籤|先驗|輸出",
                    help="--stage compare 的體型組，可重複指定。"
                         "先驗可寫 pkl:<路徑> 以自 betas 即時生成")
    ap.add_argument("--out", help="--stage compare 的 Markdown 輸出路徑")
    args = ap.parse_args()

    if args.stage == "list":
        stage_list(args)          # 純檔案掃描，不需幾何運算的相依
    elif args.stage == "measure":
        if not args.mesh:
            sys.exit("[錯誤] --stage measure 需要 --mesh")
        preflight(need_smplx=str(args.mesh).startswith("pkl:"))
        stage_measure(args)
    else:
        need = any(p[1].startswith("pkl:") or p[2].startswith("pkl:")
                   for p in args.pair)
        preflight(need_smplx=need)
        stage_compare(args)


if __name__ == "__main__":
    main()