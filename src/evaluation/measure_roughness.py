r"""
measure_roughness.py —— 網格粗糙度的正規化量測（補齊論文表 4-2）

它修正什麼
==========
表 4-2 目前以「頂點拉普拉斯座標的平均值」比較兩組拉普拉斯平滑權重，得到
「粗糙度下降 21.7%」。該數字**不可直接採信**，因為兩組網格的頂點數不同
（68,118 對 73,514）：頂點越密，相鄰頂點間距越小，拉普拉斯座標的絕對值
本就偏小。換言之，21.7% 之中有一部分只是密度差異，不是平滑的效果。

正規化的做法是除以平均邊長：

    R = mean_v || v - mean(neighbors(v)) ||  /  mean_edge_length

如此 R 成為無量綱量，可跨不同密度的網格比較。

執行後可能出現三種結果，都要照實寫：
  * 正規化後降幅仍接近 21.7% —— 原結論成立，把 21.7% 換成正規化數值即可
  * 正規化後降幅明顯縮小 —— 原數字高估了平滑的效果，正文須改寫
  * 正規化後反而上升 —— 原結論方向錯誤，須改寫為「提高權重未降低粗糙度」

用法
====
  # 1) 先依頂點數在既有輸出中定位那兩個網格
  #    表 4-2 記錄的頂點數為 68,118（權重 40）與 73,514（權重 100）
  python measure_roughness.py --stage scan --target 68118 --target 73514

  # 2) 量測並產出可直接貼入論文的表格
  python measure_roughness.py --mesh lap40.obj --label "40（預設）" \
                              --mesh lap100.obj --label "100"

  # 只量一個
  python measure_roughness.py --mesh some.obj

環境：chupa（需 trimesh、numpy、scipy）
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
    from scipy.sparse import coo_matrix
except ImportError:
    sys.exit("[錯誤] 需要 trimesh 與 scipy。請於 chupa 環境執行。")


def roughness(mesh):
    r"""回傳 (原始粗糙度, 平均邊長, 正規化粗糙度, 頂點數)。

    原始粗糙度即論文表 4-2 現有欄位的定義：頂點與其一階鄰域重心之距離，
    對全體頂點取平均。以稀疏矩陣一次算完，避免逐頂點的 Python 迴圈。
    """
    V = np.asarray(mesh.vertices, np.float64)
    E = np.asarray(mesh.edges_unique, np.int64)
    n = len(V)

    # 對稱的鄰接矩陣：每條邊貢獻兩個方向
    rows = np.concatenate([E[:, 0], E[:, 1]])
    cols = np.concatenate([E[:, 1], E[:, 0]])
    data = np.ones(len(rows))
    A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.maximum(deg, 1.0)
    centroid = A.dot(V) / deg_safe[:, None]

    lap = np.linalg.norm(V - centroid, axis=1)
    lap = lap[deg > 0]            # 孤立頂點無鄰域，排除
    raw = float(lap.mean())

    elen = float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean())
    return raw, elen, raw / elen if elen > 0 else float("nan"), n


def count_verts_fast(path):
    r"""不完整載入網格，僅計算 OBJ 的頂點數。

    以 trimesh 載入 1,600 個網格需時甚久，而尋找目標網格只需頂點數。
    OBJ 為文字格式，逐行計數 `v ` 開頭者即可，成本為單純的檔案讀取。
    """
    n = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                if line[:2] == b"v ":
                    n += 1
    except OSError:
        return None
    return n


def stage_scan(args):
    r"""掃描網格並依頂點數尋找目標。

    表 4-2 的兩組網格分別為 68,118 與 73,514 個頂點，可用此作為指紋
    在既有輸出中定位；論文未記錄其路徑，僅記錄了頂點數。
    """
    root = Path(os.path.expanduser(args.root))
    if not root.exists():
        sys.exit(f"[錯誤] 找不到目錄：{root}")

    targets = set(args.target)
    print(f"掃描 {root}（尋找頂點數 {sorted(targets) if targets else '全部'}）\n")

    cands = [p for p in root.glob("**/*.obj")
             if p.stat().st_size > 1e6 and "nds_dual" not in str(p)]
    print(f"候選網格 {len(cands)} 個，計算頂點數中...\n")

    hits, allrows = [], []
    for p in cands:
        n = count_verts_fast(p)
        if n is None:
            continue
        allrows.append((n, p))
        if not targets:
            continue
        for t in targets:
            if abs(n - t) <= args.tolerance:
                hits.append((n, t, p))

    if targets:
        if hits:
            print(f"找到 {len(hits)} 個符合（容差 ±{args.tolerance}）：\n")
            for n, t, p in sorted(hits):
                print(f"  {n:>8,} 頂點（目標 {t:,}）  {p.relative_to(root)}")
            print("\n接著以 --mesh 指定這兩個檔案執行量測。")
        else:
            print("[未找到] 目標頂點數的網格不存在於此目錄。\n")
            print("最接近的十個：\n")
            for t in sorted(targets):
                near = sorted(allrows, key=lambda r: abs(r[0] - t))[:5]
                print(f"  目標 {t:,}：")
                for n, p in near:
                    print(f"    {n:>8,}  {p.relative_to(root)}")
            print("\n若確實已不存在，重新生成兩個網格即可——依表 4-15，")
            print("單次完整生成約 10.9 分鐘，兩個約 22 分鐘，成本已不高。")
            print("生成時分別將 Chupa 的拉普拉斯平滑權重設為 40 與 100。")
    else:
        allrows.sort(key=lambda r: -r[0])
        for n, p in allrows[:40]:
            print(f"  {n:>8,}  {p.relative_to(root)}")


def main():
    ap = argparse.ArgumentParser(description="網格粗糙度的正規化量測")
    ap.add_argument("--stage", choices=["measure", "scan"], default="measure")
    ap.add_argument("--mesh", action="append", default=[])
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--root", default="~/projects/chupa",
                    help="--stage scan 的掃描起點")
    ap.add_argument("--target", action="append", type=int,
                    default=[], help="--stage scan 要尋找的頂點數，可重複")
    ap.add_argument("--tolerance", type=int, default=50,
                    help="頂點數比對的容差")
    args = ap.parse_args()

    if args.stage == "scan":
        stage_scan(args)
        return
    if not args.mesh:
        sys.exit("[錯誤] --stage measure 需要至少一個 --mesh")

    labels = args.label + [Path(m).stem for m in args.mesh[len(args.label):]]
    rows = []
    for path, label in zip(args.mesh, labels):
        m = trimesh.load(path, force="mesh", process=False)
        raw, elen, norm, n = roughness(m)
        rows.append((label, n, raw, elen, norm))
        print(f"[OK] {label}: {n:,} 頂點  原始 {raw:.6f}  "
              f"平均邊長 {elen:.6f}  正規化 {norm:.6f}")

    print("\n可直接貼入論文表 4-2：\n")
    print("| 拉普拉斯權重 | 頂點數 | 粗糙度（平均拉普拉斯座標） | 正規化粗糙度 |")
    print("|---|---|---|---|")
    for label, n, raw, elen, norm in rows:
        print(f"| {label} | {n:,} | {raw:.6f} | {norm:.6f} |")

    if len(rows) == 2:
        (_, _, r0, _, n0), (_, _, r1, _, n1) = rows
        d_raw = (r0 - r1) / r0 * 100 if r0 else float("nan")
        d_nrm = (n0 - n1) / n0 * 100 if n0 else float("nan")
        print(f"\n未正規化的降幅：{d_raw:+.1f}%（論文現行數字 21.7%）")
        print(f"正規化後的降幅：{d_nrm:+.1f}%  ← 應以此改寫正文")
        gap = abs(d_raw - d_nrm)
        if gap > 5:
            print(f"\n[注意] 兩者相差 {gap:.1f} 個百分點，"
                  f"代表原數字有相當比例來自頂點密度差異而非平滑效果，"
                  f"正文的敘述須一併修改，不能只換數字。")
        if d_nrm < 0:
            print("\n[注意] 正規化後粗糙度反而上升——原結論方向錯誤，"
                  "表 4-2 的段落須改寫為「提高平滑權重並未降低粗糙度」。")


if __name__ == "__main__":
    main()