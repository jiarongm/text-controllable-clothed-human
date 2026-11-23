r"""
run_factor_sweep.py

文字引導參數的單因子掃描：分離 $s_T$、採樣步數、重採樣次數三者的作用。

背景：現行的 2×2 消融同時改動 $s_T$ 與採樣步數，故無法將「表面細節的差異」
歸因於其中任一者。本腳本以單因子掃描修正此一混淆。

設計要點（詳見「實驗重新設計_參數掃描.md」）：

  * 三軸共用同一基準點（sT=7.5, steps=100, resample=0），該點只生成一次、
    三軸共用。若各軸分別生成，隨機性會使同一設定在三張圖上呈現不同數值。

  * 軸 A、B 以第一階段輸出判定類型；軸 C 必須以 resample 之後的輸出判定
    —— resample 的作用發生在第一階段之後，若沿用第一階段判定，該軸會是
    一條沒有意義的水平線。

  * 每個條件需多個樣本方能構成「正確率」。本腳本預設 3 提示 × 3 種子。

  * 網格粗糙度需完整雕刻（約 103 分鐘／次），故僅於子集量測；全條件則以
    法向量圖的高頻能量作為代理指標，並以子集驗證兩者的相關性。

────────────────────────────────────────────────────────────────
使用流程
────────────────────────────────────────────────────────────────

  # 0) 先確認接線（不執行生成）
  python run_factor_sweep.py --stage plan --out sweep_out

  # 0.5) 探測輸出檔名 —— 跑一次生成並列出其產生的所有檔案
  #      據此設定 NORMAL_MAP_GLOB。此步驟不可略過。
  python run_factor_sweep.py --stage probe --out sweep_out

  # 1) 生成法向量圖（全部條件）
  python run_factor_sweep.py --stage generate --out sweep_out

  # 2) 產生人工判定用的表單與對照圖
  python run_factor_sweep.py --stage sheet --out sweep_out
  #    → 開啟 sweep_out/judge/ 內的對照圖，填寫 sweep_out/judgement.csv
  #      的 correct 欄位（1=類型正確、0=不正確）

  # 3) 子集的完整網格生成（約 12 小時，建議掛背景）
  nohup python run_factor_sweep.py --stage mesh --out sweep_out > mesh.log 2>&1 &

  # 4) 分析並繪圖
  python run_factor_sweep.py --stage analyze --out sweep_out

────────────────────────────────────────────────────────────────
須先確認的整合點
────────────────────────────────────────────────────────────────

本腳本以命令列參數驅動 Chupa。下列三項須依你的實作確認並於下方常數區調整：

  1. CHUPA_ARG_* ：各參數於 Chupa 設定中的實際鍵名
  2. STAGE1_ONLY ：是否有「只跑到第一階段就停」的旗標。若無，法向量圖仍會
     產生於中間輸出，但每次生成會付出完整雕刻的成本 —— 此時強烈建議先為
     Chupa 加上該旗標，否則步驟 1 的成本會由秒級暴增為小時級。
  3. NORMAL_MAP_GLOB ：第一階段與 resample 後之法向量圖的輸出路徑樣式
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2


# ============================================================
# 整合點（依你的 Chupa 實作調整）
# ============================================================

CHUPA_PROJECT_DIR = os.path.expanduser("~/projects/chupa")
CHUPA_PYTHON = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
CHUPA_SCRIPT = "scripts/chupa.py"
CHUPA_CONFIG = "configs/text.yaml"
SUBJECT = "rp_emily_apose_001"

# 各參數在 Chupa 設定中的鍵名（已對照 configs/text.yaml 確認）
CHUPA_ARG_PROMPT = "chupa.prompt"
CHUPA_ARG_SEED = "chupa.seed"
CHUPA_ARG_ST = "chupa.text_cfg_scale"          # 文字引導強度 s_T（預設 7.5）
CHUPA_ARG_STEPS = "chupa.text_steps"           # 文字擴散採樣步數（預設 100）
CHUPA_ARG_RESAMPLE = "chupa.n_resample_from_text"   # 文字路徑的重採樣（預設 5）
CHUPA_ARG_RESAMPLE_REFINE = "chupa.n_resample"      # 精修路徑的重採樣（預設 2）

# 軸 C 只掃文字路徑的重採樣。精修路徑的重採樣發生於網格雕刻前，
# 而本掃描僅生成法向量圖、不進入雕刻，故將其固定為 0 以排除干擾。
FIX_REFINE_RESAMPLE = 0

# 只跑到第一階段就停的旗標。Chupa 原始碼中並無此旗標，
# 故預設改採「監看輸出檔案、出現後即終止行程」的方式（見 EARLY_STOP）。
# 若你已為 Chupa 加上該旗標，將其填入此處即可改走乾淨路徑。
STAGE1_ONLY = None

# 未提供 STAGE1_ONLY 時的替代方案：輪詢輸出目錄，法向量圖一出現即終止行程。
# 如此雕刻階段根本不會開始，成本與加旗標相同，且不需修改 Chupa 原始碼。
EARLY_STOP = True
EARLY_STOP_POLL = 2.0      # 輪詢間隔（秒）
EARLY_STOP_TIMEOUT = 900   # 單次生成的上限（秒），逾時視為失敗

# 法向量圖的輸出檔名樣式。已由 --stage probe 確認：
#   normal_F/000.png  ← 正面法向量圖（第一階段輸出，判定用）
#   normal_F/180.png  ← 背面，由雙面補全階段產生，在 000 之後
#   T_normal_F/*.png  ← SMPL-X 先驗渲染（36 個角度），非生成結果，須排除
#
# 目錄名稱寫進樣式即可自然排除先驗：pathlib 的 rglob 要求路徑組件完全相符，
# 故 "normal_F" 不會匹配到 "T_normal_F"。PRIOR_MARKERS 作為第二道保險。
#
# 只監看 000.png 是刻意的：它一出現即代表第一階段完成，此時終止行程，
# 雙面補全與網格雕刻根本不會開始，單次成本從約 103 分鐘降至約一分鐘。
NORMAL_MAP_GLOB = "normal_F/000.png"           # ← 已由 --stage probe 確認

# 須排除的路徑片段。依 Chupa 的命名慣例，`T_` 前綴代表 SMPL-X 先驗的渲染
# ——那是擴散模型的「輸入條件」而非生成結果。流程會先以 10 度為間隔渲染
# 36 個角度的先驗，此步驟遠早於生成階段；若不排除，監看機制會被這批輸出
# 誤觸而在生成尚未開始時就終止行程。
PRIOR_MARKERS = ("T_normal", "/T_")


# ============================================================
# 實驗設計
# ============================================================

BASE = {"st": 7.5, "steps": 100, "resample": 0}

AXES = {
    "A_guidance": ("st",       [2.0, 4.0, 6.0, 7.5, 10.0]),
    "B_steps":    ("steps",    [20, 40, 60, 80, 100]),
    "C_resample": ("resample", [0, 1, 2, 3, 5]),
}

# 原設計假設軸 C 需要另一個判定點（resample 之後的輸出），因為重採樣的效應
# 發生在第一階段之後。probe 結果顯示此假設不成立，三軸可共用同一判定點：
#
#   chupa.n_resample_from_text 是在第一階段的取樣迴圈「之內」執行的，
#   而 normal_F/000.png 於該迴圈結束後才寫出 —— 即這張圖本身就是
#   重採樣後的最終結果。probe 也確認 normal_F/ 下並無各次重採樣的分開存檔
#   （只有 000 與 180 兩張），故不存在「重採樣前」的版本可供比較。
#
# 這反而消除了原設計中「三軸判定點不一致」的弱點：三軸皆判定 000.png。
# 對應地，論文 4.2.3 節原本說明軸 C 判定點不同的段落須一併修正。
#
# 注意：此處變動的是文字路徑的重採樣（n_resample_from_text）。精修路徑的
# 重採樣（chupa.n_resample）作用於雕刻之後，而本掃描在第一階段即終止，
# 故其對本實驗無影響；為避免組態語意含混，全程固定為 0。
JUDGE_AFTER_RESAMPLE = set()

PROMPTS = [
    ("tshirt_shorts", "a person wearing a t-shirt and shorts"),
    ("long_dress",    "a woman wearing a long dress"),
    ("sleeveless",    "a person wearing a sleeveless top and long pants"),
]
SEEDS = [42, 7, 123]

# 完整網格生成的子集：每軸取最低／中間／最高，共用基準點不重複
MESH_SUBSET_SEED = 42
MESH_SUBSET_PROMPT = "tshirt_shorts"


def cond_key(st, steps, resample):
    return f"st{st:g}_steps{steps}_rs{resample}"


def enumerate_conditions():
    """回傳 {條件鍵: {st, steps, resample, axes:[所屬軸]}}，共用點只出現一次。"""
    conds = {}
    for axis, (factor, values) in AXES.items():
        for v in values:
            c = dict(BASE)
            c[factor] = v
            k = cond_key(c["st"], c["steps"], c["resample"])
            if k not in conds:
                conds[k] = {**c, "axes": []}
            conds[k]["axes"].append((axis, factor, v))
    return conds


def mesh_subset(conds):
    """每軸取最低／中間／最高三點。"""
    keys = []
    for axis, (factor, values) in AXES.items():
        for v in (values[0], values[len(values) // 2], values[-1]):
            c = dict(BASE); c[factor] = v
            k = cond_key(c["st"], c["steps"], c["resample"])
            if k not in keys:
                keys.append(k)
    return keys


# ============================================================
# 生成
# ============================================================

def build_cmd(c, prompt, seed, stage1_only):
    cmd = [CHUPA_PYTHON, CHUPA_SCRIPT, CHUPA_CONFIG,
           f"dataset.subject={SUBJECT}",
           f"{CHUPA_ARG_PROMPT}={prompt}",
           f"{CHUPA_ARG_SEED}={seed}",
           f"{CHUPA_ARG_ST}={c['st']}",
           f"{CHUPA_ARG_STEPS}={c['steps']}",
           f"{CHUPA_ARG_RESAMPLE}={c['resample']}",
           f"{CHUPA_ARG_RESAMPLE_REFINE}={FIX_REFINE_RESAMPLE}"]
    if stage1_only and STAGE1_ONLY:
        cmd.append(STAGE1_ONLY)
    return cmd


def is_prior_render(path):
    """判斷是否為 SMPL-X 先驗的渲染輸出（擴散模型的輸入條件，非生成結果）。"""
    s = str(path)
    return any(m in s for m in PRIOR_MARKERS)


def run_until_normal_map(cmd, timeout=EARLY_STOP_TIMEOUT):
    """執行生成，於法向量圖寫出後立即終止行程。

    Chupa 沒有「只跑第一階段」的旗標，而完整流程含網格雕刻約需 103 分鐘。
    本函式輪詢輸出目錄，一旦偵測到新的法向量圖即終止行程 —— 雕刻階段
    根本不會開始，成本與加旗標相同，且不需修改 Chupa 原始碼。

    回傳寫出的法向量圖路徑；逾時或失敗回傳 None。
    """
    import signal
    root = Path(CHUPA_PROJECT_DIR) / "results"
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=CHUPA_PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    found = None
    try:
        while True:
            if proc.poll() is not None:
                # 行程已自行結束（可能是失敗，也可能整段跑完）
                break
            cands = [p for p in root.rglob(NORMAL_MAP_GLOB)
                     if p.stat().st_mtime >= t0 and not is_prior_render(p)]
            if cands:
                # 稍候片刻確保檔案寫入完成，再終止
                time.sleep(1.0)
                found = max(cands, key=lambda p: p.stat().st_mtime)
                break
            if time.time() - t0 > timeout:
                print(f"    [逾時] 超過 {timeout} 秒仍未產生法向量圖")
                break
            time.sleep(EARLY_STOP_POLL)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    if found is None:
        cands = [p for p in root.rglob(NORMAL_MAP_GLOB)
                 if p.stat().st_mtime >= t0]
        found = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
    return found


def stage_probe(args):
    """跑一次生成並列出其產生的所有檔案，用以確認法向量圖的實際檔名。

    此步驟必須先做：NORMAL_MAP_GLOB 若設錯，監看機制會一直等不到檔案而逾時，
    或抓到錯誤階段的輸出（例如抓到 resample 之前的而非之後的）。
    """
    c = dict(BASE)
    cmd = build_cmd(c, PROMPTS[0][1], SEEDS[0], stage1_only=True)
    print("執行一次生成以探測輸出檔名（將於偵測到法向量圖後終止）")
    print("  $ " + " ".join(cmd) + "\n")

    root = Path(CHUPA_PROJECT_DIR) / "results"
    t0 = time.time()
    import signal
    proc = subprocess.Popen(cmd, cwd=CHUPA_PROJECT_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    seen, settle_until = [], None
    try:
        while time.time() - t0 < EARLY_STOP_TIMEOUT:
            if proc.poll() is not None:
                print("\n[提示] 生成行程已自行結束")
                break
            new = [p for p in root.rglob("*")
                   if p.is_file() and p.stat().st_mtime >= t0 and p not in seen]
            for p in sorted(new, key=lambda q: q.stat().st_mtime):
                seen.append(p)
                mark = "  (先驗渲染)" if is_prior_render(p) else "  ← 生成輸出"
                print(f"  [{p.stat().st_mtime - t0:6.1f}s] {p.relative_to(root)}{mark}")

            # 只有「非先驗渲染」的輸出才視為生成階段的產物。
            # 先驗渲染（T_ 前綴）為擴散模型的輸入條件，早於生成階段產生，
            # 若以其為觸發條件會在生成尚未開始時就終止行程。
            gen = [p for p in seen
                   if p.suffix.lower() == ".png" and not is_prior_render(p)]
            if gen and settle_until is None:
                settle_until = time.time() + 45   # 再收集 45 秒同批輸出
                print(f"\n  → 偵測到生成輸出，再收集 45 秒後結束\n")
            if settle_until and time.time() > settle_until:
                break
            time.sleep(2)
        else:
            print(f"\n[逾時] {EARLY_STOP_TIMEOUT} 秒內未偵測到生成輸出")
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=15)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    prior = [p for p in seen if p.suffix.lower() == ".png" and is_prior_render(p)]
    gen = [p for p in seen if p.suffix.lower() == ".png" and not is_prior_render(p)]
    other = [p for p in seen if p.suffix.lower() != ".png"]

    print("\n" + "=" * 68)
    print(f"先驗渲染（已排除）：{len(prior)} 個")
    if prior:
        dirs = sorted({str(p.parent.relative_to(root)) for p in prior})
        for d in dirs:
            print(f"  {d}/")
    print(f"\n生成輸出：{len(gen)} 個")
    for p in gen:
        print(f"  {p.relative_to(root)}")
    if other:
        print(f"\n其他檔案：{len(other)} 個")
        for p in other[:10]:
            print(f"  {p.relative_to(root)}")
    print("=" * 68)

    if not gen:
        print("\n未捕捉到生成輸出。可能原因：")
        print("  1. 逾時過短 —— 提高 EARLY_STOP_TIMEOUT 後重試")
        print("  2. 生成結果不以 png 存放 —— 檢查上方「其他檔案」")
        print("  3. PRIOR_MARKERS 過於寬鬆，把生成輸出也排除了")
    else:
        print("\n請自「生成輸出」中辨識出正面法向量圖，據此設定 NORMAL_MAP_GLOB。")
        print("若其中含多張編號不同的法向量圖，代表各次重採樣另存為不同檔案；")
        print("此時軸 C 應明確取用最後一張（重採樣後的最終結果）。")


def stage_plan(args):
    conds = enumerate_conditions()
    subset = mesh_subset(conds)
    n_samples = len(PROMPTS) * len(SEEDS)

    print("=" * 68)
    print("單因子掃描：執行計畫")
    print("=" * 68)
    print(f"\n唯一條件數：{len(conds)}（三軸共 15 點，扣除共用基準點 2 次重複）")
    print(f"每條件樣本：{len(PROMPTS)} 提示 × {len(SEEDS)} 種子 = {n_samples}")
    print(f"法向量圖總數：{len(conds) * n_samples}")
    print(f"完整網格子集：{len(subset)} 個條件 × 1 樣本"
          f"（約 {len(subset) * 103 / 60:.1f} 小時）\n")

    print(f"{'條件':<26}{'sT':>6}{'steps':>7}{'rs':>5}  所屬軸")
    for k, c in conds.items():
        axes = ", ".join(f"{a}={v:g}" for a, _, v in c["axes"])
        mark = " ★網格" if k in subset else ""
        print(f"{k:<26}{c['st']:>6g}{c['steps']:>7}{c['resample']:>5}  {axes}{mark}")

    print("\n判定點：")
    for axis in AXES:
        where = "resample 之後" if axis in JUDGE_AFTER_RESAMPLE else "第一階段"
        print(f"  {axis:<14} → {where}")

    print("\n範例指令（第一個條件、第一個提示）：")
    k0 = list(conds)[0]
    print("  $ " + " ".join(build_cmd(conds[k0], PROMPTS[0][1], SEEDS[0], True)))

    if not STAGE1_ONLY:
        print("\n[警告] STAGE1_ONLY 未設定。每次生成將付出完整雕刻的成本，")
        print("       全條件掃描（117 次）約需 200 小時，不可行。")
        print("       請先為 Chupa 加上「只跑第一階段」的旗標。")


def stage_generate(args):
    conds = enumerate_conditions()
    out = Path(args.out) / "normals"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    total = len(conds) * len(PROMPTS) * len(SEEDS)
    n, t0 = 0, time.time()
    for k, c in conds.items():
        for pname, prompt in PROMPTS:
            for seed in SEEDS:
                n += 1
                tag = f"{k}__{pname}__s{seed}"
                dst = out / f"{tag}.png"
                if dst.exists() and not args.force:
                    print(f"[{n}/{total}] {tag} 已存在，跳過")
                    manifest.append({"tag": tag, "cond": k, "prompt": pname,
                                     "seed": seed, "png": str(dst), **c})
                    continue

                cmd = build_cmd(c, prompt, seed, stage1_only=True)
                print(f"[{n}/{total}] {tag}")
                if args.dry_run:
                    print("  $ " + " ".join(cmd))
                    continue

                t = time.time()
                if STAGE1_ONLY:
                    r = subprocess.run(cmd, cwd=CHUPA_PROJECT_DIR,
                                       capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  ✗ 失敗：{r.stderr[-600:]}")
                        continue
                    src = _find_normal_map(t)
                else:
                    src = run_until_normal_map(cmd)

                if src is None:
                    print("  ✗ 找不到法向量圖輸出，請先執行 --stage probe "
                          "確認 NORMAL_MAP_GLOB")
                    continue
                cv2.imwrite(str(dst), cv2.imread(str(src)))
                print(f"    {(time.time() - t):.0f} 秒")
                manifest.append({"tag": tag, "cond": k, "prompt": pname,
                                 "seed": seed, "png": str(dst), **c})

    (Path(args.out) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成 {len(manifest)} 筆，耗時 {(time.time() - t0)/60:.1f} 分鐘")


def _find_normal_map(since_ts):
    """在 Chupa 輸出中找出這次生成的法向量圖。

    軸 C 的判定點在 resample 之後，而 Chupa 的重採樣係就地更新同一張圖，
    故此處一律取「最後寫出」者 —— 對 resample=0 的條件即為第一階段輸出，
    對 resample>0 的條件則為重採樣後的結果，恰為各軸所需的判定對象。
    """
    root = Path(CHUPA_PROJECT_DIR) / "results"
    cands = [p for p in root.rglob(NORMAL_MAP_GLOB) if p.stat().st_mtime >= since_ts]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


# ============================================================
# 指標
# ============================================================

def high_freq_energy(png_path, sigma=3.0):
    """法向量圖的高頻能量，作為表面細節豐富度的代理指標。

    於人體遮罩內計算「原圖與高斯模糊之差」的均方根。遮罩以非背景像素近似
    —— 背景為均勻色，其高頻能量為零，若不排除會依人體佔畫面比例稀釋數值。
    """
    im = cv2.imread(str(png_path)).astype(np.float32)
    if im is None:
        return None
    blur = cv2.GaussianBlur(im, (0, 0), sigma)
    diff = im - blur
    # 背景近似：與四角中位色相近者視為背景
    corners = np.concatenate([im[:8, :8].reshape(-1, 3), im[:8, -8:].reshape(-1, 3),
                              im[-8:, :8].reshape(-1, 3), im[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    mask = np.linalg.norm(im - bg, axis=2) > 12
    if mask.sum() < 100:
        mask = np.ones(im.shape[:2], bool)
    return float(np.sqrt((np.sum(diff ** 2, axis=2)[mask]).mean()))


def mesh_roughness(obj_path):
    """網格的正規化粗糙度：平均拉普拉斯座標長度 ÷ 平均邊長。

    正規化為必要 —— 不同生成之網格密度有別，未正規化的絕對值不可跨條件比較。
    """
    import trimesh
    m = trimesh.load(str(obj_path), force="mesh", process=False)
    v = np.asarray(m.vertices, np.float64)
    f = np.asarray(m.faces, np.int64)

    from scipy.sparse import coo_matrix
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    e = np.vstack([e, e[:, ::-1]])
    V = len(v)
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(V, V)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1
    lap = v - (A @ v) / deg[:, None]
    mean_len = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean()
    return float(np.linalg.norm(lap, axis=1).mean() / max(mean_len, 1e-12))


def wilson(k, n, z=1.96):
    """Wilson 信賴區間 —— 小樣本下較常態近似合適。"""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ============================================================
# 判定表單
# ============================================================

def stage_sheet(args):
    out = Path(args.out)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jdir = out / "judge"
    jdir.mkdir(exist_ok=True)

    # 每個條件一張對照圖，9 個樣本排成 3×3
    by_cond = {}
    for m in man:
        by_cond.setdefault(m["cond"], []).append(m)

    for cond, items in by_cond.items():
        items = sorted(items, key=lambda m: (m["prompt"], m["seed"]))
        tiles = []
        for m in items:
            im = cv2.imread(m["png"])
            if im is None:
                continue
            im = cv2.resize(im, (320, 320))
            band = np.full((28, 320, 3), 255, np.uint8)
            cv2.putText(band, f"{m['prompt'][:14]} s{m['seed']}", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([band, im]))
        if not tiles:
            continue
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)),
                       constant_values=255) for r in rows]
        cv2.imwrite(str(jdir / f"{cond}.png"), np.vstack(rows))

    csv_path = out / "judgement.csv"
    if csv_path.exists() and not args.force:
        print(f"{csv_path} 已存在，未覆寫（要重建請加 --force）")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tag", "cond", "prompt", "seed", "st", "steps",
                        "resample", "correct"])
            for m in sorted(man, key=lambda m: (m["cond"], m["prompt"], m["seed"])):
                w.writerow([m["tag"], m["cond"], m["prompt"], m["seed"],
                            m["st"], m["steps"], m["resample"], ""])
        print(f"已建立 {csv_path}")

    print(f"\n對照圖：{jdir}")
    print("請逐一檢視各條件的對照圖，於 judgement.csv 的 correct 欄填入 1 或 0。")
    print("判定準則：生成之服裝幾何是否符合提示的關鍵特徵")
    print("  tshirt_shorts → 是否為短袖且短褲（手臂與小腿裸露）")
    print("  long_dress    → 是否呈現兩腿間連續的裙襬")
    print("  sleeveless    → 是否為無袖且長褲")


# ============================================================
# 完整網格子集
# ============================================================

def stage_mesh(args):
    conds = enumerate_conditions()
    subset = mesh_subset(conds)
    prompt = dict(PROMPTS)[MESH_SUBSET_PROMPT]
    out = Path(args.out) / "meshes"
    out.mkdir(parents=True, exist_ok=True)
    rec = []

    print(f"完整網格子集：{len(subset)} 個條件，"
          f"預估 {len(subset) * 103 / 60:.1f} 小時\n")
    for i, k in enumerate(subset, 1):
        c = conds[k]
        dst = out / f"{k}.obj"
        if dst.exists() and not args.force:
            print(f"[{i}/{len(subset)}] {k} 已存在，跳過")
        else:
            print(f"[{i}/{len(subset)}] {k} 生成中 ...")
            if args.dry_run:
                print("  $ " + " ".join(build_cmd(c, prompt, MESH_SUBSET_SEED, False)))
                continue
            t = time.time()
            r = subprocess.run(build_cmd(c, prompt, MESH_SUBSET_SEED, False),
                               cwd=CHUPA_PROJECT_DIR, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  ✗ 失敗：{r.stderr[-600:]}")
                continue
            cands = [p for p in (Path(CHUPA_PROJECT_DIR) / "results").rglob(
                "mesh_refine.obj") if p.stat().st_mtime >= t]
            if not cands:
                print("  ✗ 找不到輸出網格")
                continue
            import shutil
            shutil.copy(max(cands, key=lambda p: p.stat().st_mtime), dst)
            print(f"  完成（{(time.time() - t)/60:.1f} 分鐘）")

        if dst.exists():
            rec.append({"cond": k, "obj": str(dst),
                        "roughness": mesh_roughness(dst), **{
                            kk: c[kk] for kk in ("st", "steps", "resample")}})

    (Path(args.out) / "mesh_roughness.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n已寫出 mesh_roughness.json（{len(rec)} 筆）")


# ============================================================
# 分析與繪圖
# ============================================================

def stage_analyze(args):
    out = Path(args.out)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    # 判定結果
    judged = {}
    csv_path = out / "judgement.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["correct"].strip() in ("0", "1"):
                    judged[row["tag"]] = int(row["correct"])
    n_missing = len(man) - len(judged)
    if n_missing:
        print(f"[警告] {n_missing} 筆尚未判定，正確率僅以已判定者計算")

    # 高頻能量
    hf = {}
    for m in man:
        e = high_freq_energy(m["png"])
        if e is not None:
            hf[m["tag"]] = e

    # 網格粗糙度（子集）
    mesh = {}
    mp = out / "mesh_roughness.json"
    if mp.exists():
        for r in json.loads(mp.read_text(encoding="utf-8")):
            mesh[r["cond"]] = r["roughness"]

    conds = enumerate_conditions()
    results = {}
    for axis, (factor, values) in AXES.items():
        rows = []
        for v in values:
            c = dict(BASE); c[factor] = v
            k = cond_key(c["st"], c["steps"], c["resample"])
            tags = [m["tag"] for m in man if m["cond"] == k]
            kk = [judged[t] for t in tags if t in judged]
            p, lo, hi = wilson(sum(kk), len(kk))
            ee = [hf[t] for t in tags if t in hf]
            rows.append({"value": v, "n": len(kk), "correct": p,
                         "ci_lo": lo, "ci_hi": hi,
                         "hf_mean": float(np.mean(ee)) if ee else None,
                         "hf_std": float(np.std(ee)) if ee else None,
                         "mesh_roughness": mesh.get(k)})
        results[axis] = {"factor": factor, "rows": rows}

    (out / "sweep_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # 代理指標的有效性
    #
    # 必須分軸計算，不可全部混算。軸 C 改變 resample，其效應是「服裝被抹回
    # 長袖長褲」——布料覆蓋面積變大使高頻能量上升，但重採樣本身讓網格更平滑，
    # 兩個指標因不同機制被拉往相反方向。把軸 C 混進來會使相關係數趨近於零，
    # 並得出「代理指標無效」的錯誤結論。
    #
    # 代理指標的前提本就是「被比較者的服裝內容相同」，只有軸 A 與軸 B 滿足。
    def _corr(rows):
        pairs = [(r["hf_mean"], r["mesh_roughness"]) for r in rows
                 if r["hf_mean"] is not None and r["mesh_roughness"] is not None]
        # 三軸共用同一基準點，混算會重複計入而虛增樣本數
        uniq = sorted(set(pairs))
        if len(uniq) < 3:
            return None, len(uniq)
        a, b = np.array(uniq).T
        return float(np.corrcoef(a, b)[0, 1]), len(uniq)

    print("\n代理指標驗證：法向量圖高頻能量 vs 網格正規化粗糙度")
    per_axis = {}
    for axis, res in results.items():
        c, n = _corr(res["rows"])
        per_axis[axis] = {"r": c, "n": n}
        if c is not None:
            print(f"  {axis:14s} r = {c:+.3f}（{n} 點）")
        else:
            print(f"  {axis:14s} 配對點不足（{n} 點）")

    valid_rows = [r for a, res in results.items() if a != "C_resample"
                  for r in res["rows"]]
    corr, n_valid = _corr(valid_rows)
    all_rows = [r for res in results.values() for r in res["rows"]]
    corr_all, n_all = _corr(all_rows)

    if corr is not None:
        print(f"\n  軸 A+B（服裝內容固定）：r = {corr:+.3f}（{n_valid} 點）  ← 應採用此值")
    if corr_all is not None:
        print(f"  全部含軸 C：           r = {corr_all:+.3f}（{n_all} 點）  "
              f"（軸 C 之高頻能量因服裝改變而不可比，僅供對照）")

    if corr is not None:
        if abs(corr) >= 0.6:
            print("\n  [結論] 於服裝內容固定的條件下，代理指標有效，"
                  "可用高頻能量推論網格粗糙度。")
        else:
            print("\n  [結論] 即使排除軸 C，相關性仍偏低；"
                  "論文中應分別呈現兩者，勿以前者推論後者。")

    _print_tables(results)
    _plot(results, out, corr)


def _print_tables(results):
    for axis, d in results.items():
        print(f"\n{'=' * 68}\n{axis}（變動 {d['factor']}）\n{'=' * 68}")
        print(f"{'值':>8}{'n':>5}{'類型正確率':>14}{'95% CI':>20}"
              f"{'高頻能量':>12}{'網格粗糙度':>14}")
        for r in d["rows"]:
            ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
            hfm = f"{r['hf_mean']:.3f}" if r["hf_mean"] is not None else "—"
            mr = f"{r['mesh_roughness']:.5f}" if r["mesh_roughness"] is not None else "—"
            print(f"{r['value']:>8g}{r['n']:>5}{r['correct']*100:>13.1f}%"
                  f"{ci:>20}{hfm:>12}{mr:>14}")


def _plot(results, out, corr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[提示] 未安裝 matplotlib，跳過繪圖。pip install matplotlib")
        return

    titles = {"A_guidance": "Text guidance scale $s_T$",
              "B_steps": "Sampling steps",
              "C_resample": "Resample count"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (name, d) in zip(axes, results.items()):
        rows = d["rows"]
        x = [r["value"] for r in rows]
        y = [r["correct"] * 100 for r in rows]

        # 不繪製 Wilson 信賴區間。各條件共用同一組 9 個 prompt–seed 單元，
        # 故各點並非獨立樣本，依 n 計算的區間會過窄；正文亦已據此改為
        # 描述性比率（4.2.3、4.2.4 節）。此處畫出誤差棒等同以視覺重述一個
        # 已被移除的統計宣稱，故僅繪點與連線。
        ax.plot(x, y, marker="o", color="#1f77b4", label="Type correctness")
        for xi, yi, r in zip(x, y, rows):
            ax.annotate(f"{r['correct']*100:.0f}%\n(n={r['n']})",
                        (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color="#1f77b4")
        ax.set_xlabel(titles.get(name, name))
        ax.set_ylabel("Type correctness (%)", color="#1f77b4")
        ax.set_ylim(-5, 105)
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        hfm = [r["hf_mean"] for r in rows]
        if all(v is not None for v in hfm):
            hfs = [r["hf_std"] or 0 for r in rows]
            ax2.errorbar(x, hfm, yerr=hfs, marker="s", color="#d62728",
                         capsize=3, linestyle="--", label="Normal-map HF energy")
        mx = [r["value"] for r in rows if r["mesh_roughness"] is not None]
        my = [r["mesh_roughness"] for r in rows if r["mesh_roughness"] is not None]
        if mx:
            ax3 = ax2
            ax3.scatter(mx, np.array(my) * (np.nanmean(hfm) / np.mean(my))
                        if hfm and all(v is not None for v in hfm) else my,
                        marker="^", s=70, color="#2ca02c", zorder=5,
                        label="Mesh roughness (scaled)")
        ax2.set_ylabel("Surface detail", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    if corr is not None:
        fig.suptitle(f"Single-factor sweeps  (HF-energy vs mesh-roughness  r = {corr:.2f})",
                     fontsize=11)
    fig.tight_layout()
    p = Path(out) / "fig_factor_sweep.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"\n折線圖：{p}")


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["plan", "probe", "generate", "sheet", "mesh", "analyze"])
    p.add_argument("--out", default="sweep_out")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    {"plan": stage_plan, "probe": stage_probe, "generate": stage_generate,
     "sheet": stage_sheet, "mesh": stage_mesh,
     "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()