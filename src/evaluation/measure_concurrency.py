r"""
measure_concurrency.py —— 量測背景生成期間的即時更新率（GPU 資源競爭）

要回答的問題
============
論文 3.6 節主張換裝為非同步設計：即時迴圈不因等待生成而阻塞。惟三個執行
環境部署於同一工作站、共用單一張 GPU，故背景生成期間存在資源競爭。4.7 節
所報告的 21.0 FPS 係於**無背景生成**的條件下量得，因此「背景生成期間仍
維持該更新率」目前並無證據。本腳本補上該量測。

設計
====
於同一段影片、同樣的影格數上，量測兩種條件的端到端逐影格延遲：

  A  基線      ：僅執行驅動路徑
  B  競爭      ：驅動路徑執行的同時，背景執行一次 Chupa 生成
  A' 基線重測  ：再測一次 A，用以檢查熱節流或其他隨時間的漂移

**A' 是必要的。** 若 B 較 A 慢，可能是 GPU 競爭，也可能是 GPU 溫度上升
後的降頻——兩者於單次 A→B 的比較中無法分辨。若 A' 與 A 相近，則 B 的差異
方可歸因於競爭；若 A' 已明顯慢於 A，則本次量測不足以歸因，須降頻後重測。

**確認生成確實在跑。** 生成行程啟動後需數十秒載入模型，此期間 GPU 尚未
真正忙碌。若於此時開始量測，會低估競爭的影響。本腳本以 nvidia-smi 輪詢
GPU 使用率，待其持續高於門檻後才開始量測 B；若逾時仍未達到，則明確報錯
而非默默量測一段不具代表性的區間。

用法
====
  conda activate romp      # 需同時具備 romp 與 smplx（同 profile_pipeline）
  python measure_concurrency.py --video ~/projects/chupa/motion_testVideo/1.mp4

  # 只印出將執行的生成指令，不實際執行
  python measure_concurrency.py --video ... --dry-run

環境：romp（驅動量測）；生成子行程自行使用 chupa 環境
"""

import argparse
import json
import os
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PY_CHUPA = os.path.expanduser("~/miniconda3/envs/chupa/bin/python")
GEN_ARGS = [
    "scripts/chupa.py", "configs/text.yaml",
    "dataset.subject=rp_emily_apose_001",
    "chupa.prompt=a person wearing a t-shirt and shorts",
    "chupa.seed=42",
    "chupa.text_cfg_scale=7.5",
    "chupa.text_steps=100",
    "chupa.n_resample_from_text=0",
    "chupa.n_resample=0",
]

GPU_BUSY_THRESHOLD = 50      # %，視為「生成確實在用 GPU」的門檻
GPU_BUSY_CONSECUTIVE = 3     # 連續幾次輪詢達標才開始量測
GPU_WAIT_TIMEOUT = 180       # 秒；逾時即報錯


class ResourceSampler:
    r"""於量測期間背景取樣資源用量。

    **為何必要。** 若僅知道「競爭時變慢」，無從得知成因；而不同成因的緩解
    方式截然不同——顯示記憶體不足應分配獨立裝置或降低生成解析度，運算資源
    競爭應調整排程優先權，系統記憶體交換則須降低常駐記憶體。三者的判別
    必須於**劣化實際發生的當下**取樣，事後或閒置時的觀察皆無效。
    """

    def __init__(self, interval=2.0):
        import threading
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        vram = util = None
        try:
            o = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            vram, util = [int(x) for x in o.split("\n")[0].split(",")]
        except Exception:
            pass
        ram = swap = None
        try:                                  # /proc/meminfo：WSL2 亦適用
            m = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                m[k] = int(v.strip().split()[0])   # kB
            ram = (m["MemTotal"] - m["MemAvailable"]) // 1024
            swap = (m["SwapTotal"] - m["SwapFree"]) // 1024
        except Exception:
            pass
        return {"vram_mb": vram, "gpu_util": util,
                "ram_mb": ram, "swap_mb": swap}

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.rows.append(self._sample())

    def __enter__(self):
        self.rows.append(self._sample())
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def summary(self):
        def agg(key):
            v = [r[key] for r in self.rows if r.get(key) is not None]
            return {"mean": round(sum(v) / len(v), 1), "max": max(v)} if v else None
        return {"n_samples": len(self.rows), "vram_mb": agg("vram_mb"),
                "gpu_util": agg("gpu_util"), "ram_mb": agg("ram_mb"),
                "swap_mb": agg("swap_mb")}


def gpu_util():
    """回傳 GPU 使用率（%）。取不到時回傳 None 而非假設為 0。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def wait_until_busy():
    """等待生成行程真正開始使用 GPU。回傳是否成功。"""
    print(f"  等待 GPU 使用率連續 {GPU_BUSY_CONSECUTIVE} 次高於 "
          f"{GPU_BUSY_THRESHOLD}%（最多 {GPU_WAIT_TIMEOUT} 秒）...")
    t0, hits = time.time(), 0
    while time.time() - t0 < GPU_WAIT_TIMEOUT:
        u = gpu_util()
        if u is None:
            print("  [警告] 無法讀取 nvidia-smi，改以固定等待 60 秒")
            time.sleep(60)
            return True
        hits = hits + 1 if u >= GPU_BUSY_THRESHOLD else 0
        if hits >= GPU_BUSY_CONSECUTIVE:
            print(f"  GPU 使用率 {u}%，開始量測")
            return True
        time.sleep(2)
    return False


def drive_loop(video, frames, warmup, work, bind_pkl, model_dir):
    """執行驅動路徑並回傳逐影格的端到端耗時（毫秒）。

    量測範圍與 profile_pipeline.py --part endtoend 一致：
    影像前處理 → ROMP → 姿態轉換 → FK → LBS。不含光柵化與合成。
    """
    import cv2
    import romp
    import torch
    from scipy.spatial.transform import Rotation
    from profile_pipeline import fk, load_bind, build_smplx

    settings = romp.main.default_settings
    settings.calc_smpl = True
    model = romp.ROMP(settings)

    g = np.load(Path(work) / "geometry.npz")
    sw = np.load(Path(work) / "skin_weights.npz")
    verts_w = g["verts"].astype(np.float64)
    sj, swt = sw["joints"].astype(np.int64), sw["weights"].astype(np.float64)

    bind = load_bind(bind_pkl)
    # profile_pipeline.build_smplx 只回傳 (J_rest, parents)，
    # 與 drive_avatar.build_smplx 的四個回傳值不同，勿混用。
    J_rest, parents = build_smplx(model_dir, bind["betas"])
    G_bind_inv = np.linalg.inv(fk(J_rest, bind["full_pose"], parents))
    v_model = (verts_w - bind["transl"].reshape(1, 3)) / bind["scale"]
    vh = np.concatenate([v_model, np.ones((len(v_model), 1))], axis=1)
    R_CW = np.diag([1.0, -1.0, -1.0])

    cap = cv2.VideoCapture(str(video))
    times, n = [], 0
    sync = torch.cuda.synchronize if torch.cuda.is_available() else None

    while n < frames + warmup:
        ok, frame = cap.read()
        if not ok:                       # 影片較短時自動循環
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = model(rgb)
        if out is None or "smpl_thetas" not in out or len(out["smpl_thetas"]) == 0:
            continue
        theta = np.asarray(out["smpl_thetas"][0]).reshape(-1, 3)

        go = Rotation.from_matrix(
            R_CW @ Rotation.from_rotvec(theta[0]).as_matrix()).as_rotvec()
        full = np.array(bind["full_pose"], np.float64).reshape(-1, 3).copy()
        full[0] = go
        full[1:22] = theta[1:22]

        G = fk(J_rest, full.reshape(-1), parents)
        T = np.einsum("jab,jbc->jac", G, G_bind_inv)
        M = np.einsum("nk,nkab->nab", swt, T[sj])
        _ = np.einsum("nab,nb->na", M, vh)[:, :3]

        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1e3)
        n += 1

    cap.release()
    return times[warmup:]                # 丟棄暖機影格


def summarise(name, t):
    a = np.array(t, float)
    return {
        "name": name, "n": len(a),
        "mean": float(a.mean()), "std": float(a.std(ddof=1)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
        "fps": float(1000.0 / a.mean()),
    }


def show(s):
    print(f"  {s['name']:14s} n={s['n']:4d}  mean {s['mean']:7.2f} ms  "
          f"std {s['std']:6.2f}  P95 {s['p95']:7.2f}  max {s['max']:7.2f}  "
          f"→ {s['fps']:5.1f} FPS")


def main():
    ap = argparse.ArgumentParser(description="量測背景生成期間的即時更新率")
    ap.add_argument("--video", required=True)
    ap.add_argument("--work", default="demo_work/A2_navy_coat")
    ap.add_argument("--bind-pkl",
                    default="data/renderpeople/smplx/rp_emily_apose_001.pkl")
    ap.add_argument("--smplx-model-dir", default="src/smpl_related/models")
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="concurrency_out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        print("生成子行程指令：")
        print("  $ " + " ".join([PY_CHUPA] + GEN_ARGS))
        return

    if not Path(args.video).exists():
        sys.exit(f"[錯誤] 找不到影片 {args.video}")

    kw = dict(video=args.video, frames=args.frames, warmup=args.warmup,
              work=args.work, bind_pkl=args.bind_pkl,
              model_dir=args.smplx_model_dir)

    print("\n=== A 基線：僅執行驅動 ===")
    with ResourceSampler() as sa:
        a = summarise("A 基線", drive_loop(**kw))
    show(a)

    print("\n=== B 競爭：驅動 + 背景生成 ===")
    proc = subprocess.Popen([PY_CHUPA] + GEN_ARGS, cwd=ROOT,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True)   # 便於整組終止
    try:
        if not wait_until_busy():
            proc_kill(proc)
            sys.exit("[錯誤] 生成行程逾時仍未使 GPU 忙碌，本次量測作廢——"
                     "若逕行量測，所得將是生成尚在載入模型期間的數據，"
                     "會低估競爭的影響。請確認生成指令可正常執行。")
        with ResourceSampler() as sb:
            b = summarise("B 競爭", drive_loop(**kw))
    finally:
        proc_kill(proc)
    show(b)

    print("\n=== A' 基線重測：檢查漂移 ===")
    with ResourceSampler() as sa2:
        a2 = summarise("A' 重測", drive_loop(**kw))
    show(a2)

    # --- 資源用量：判別劣化的成因 ---
    print("\n資源用量（量測期間之背景取樣）")
    print(f"  {'條件':10s} {'顯示記憶體 MB':>18s} {'GPU 使用率 %':>14s} "
          f"{'系統記憶體 MB':>16s} {'置換 MB':>12s}")
    res = {}
    for tag, s in (("A 基線", sa), ("B 競爭", sb), ("A' 重測", sa2)):
        d = s.summary()
        res[tag] = d
        def f(k):
            return f"{d[k]['mean']:.0f} / {d[k]['max']}" if d[k] else "n/a"
        print(f"  {tag:10s} {f('vram_mb'):>18s} {f('gpu_util'):>14s} "
              f"{f('ram_mb'):>16s} {f('swap_mb'):>12s}")
    print("  （格式為 平均 / 最大）")

    vb = res["B 競爭"].get("vram_mb")
    sw = res["B 競爭"].get("swap_mb")
    swa = res["A 基線"].get("swap_mb")
    print("\n成因判別：")
    if vb and vb["max"] >= 11000:
        print("  → 顯示記憶體於競爭期間逼近上限，**支持記憶體不足假說**；"
              "緩解方向為分配獨立運算裝置或降低生成階段的解析度。")
    elif vb:
        print(f"  → 顯示記憶體最高僅 {vb['max']} MB，距 12 GB 尚有餘裕，"
              "**不支持顯示記憶體不足假說**。")
    if sw and swa and sw["max"] - swa["max"] > 500:
        print(f"  → 系統置換空間於競爭期間增加 {sw['max'] - swa['max']} MB，"
              "**支持系統記憶體交換假說**（WSL2 下尤須留意）。")
    elif sw is not None:
        print("  → 系統置換空間未明顯增加，不支持交換假說。")
    print("  若二者皆不支持，則劣化較可能來自運算資源的排程競爭本身"
          "（於 WSL2 之 GPU 透通下，長時間佔用的核心可能阻塞其他行程）。")

    # --- 判讀 ---
    print("\n" + "=" * 70)
    drift = (a2["mean"] - a["mean"]) / a["mean"] * 100
    delta = (b["mean"] - a["mean"]) / a["mean"] * 100
    print(f"A → A' 漂移：{drift:+.1f}%（熱節流等時間相關效應）")
    print(f"A → B  變化：{delta:+.1f}%　FPS {a['fps']:.1f} → {b['fps']:.1f}")

    if abs(drift) > 0.5 * abs(delta) and abs(delta) > 3:
        print("\n  [警告] 基線本身的漂移相對於 A→B 的差異並不小，"
              "**本次結果不足以將差異歸因於 GPU 競爭**。"
              "建議待 GPU 降溫後重測，或增加重複次數。")
    elif abs(delta) < 3:
        print("\n  結論：背景生成期間的更新率與基線相近（差異 < 3%），"
              "支持論文「非同步設計於實務上可行」之敘述。")
    else:
        print(f"\n  結論：背景生成使更新率下降 {abs(delta):.1f}%"
              f"（{a['fps']:.1f} → {b['fps']:.1f} FPS）。"
              "**論文 3.6 節之敘述須據此修正**，並重新評估預生成資產庫"
              "應定位為權宜設計或必要條件。")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "concurrency.json").write_text(json.dumps(
        {"baseline": a, "contended": b, "baseline_repeat": a2,
         "drift_pct": drift, "delta_pct": delta,
         "resources": res,
         "video": args.video, "frames": args.frames,
         "gen_cmd": " ".join(GEN_ARGS)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[完成] {out / 'concurrency.json'}")


def proc_kill(proc):
    """終止生成行程及其子行程。Chupa 會衍生子行程，僅 kill 父行程會留下孤兒。"""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
