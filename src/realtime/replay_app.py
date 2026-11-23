r"""
replay_app.py —— 以預錄影片取代攝影機，驅動同一條即時迴圈

用途有二：
  1. 繞開 WSL2 下瀏覽器無法取得攝影機權限的問題，先驗證迴圈本身可運作。
  2. 以可重現的輸入錄製 demo——**與 4.7 節的效能量測採同一種輸入**，
     該節即載明「以預錄影片而非即時攝影機為輸入，是為使效能量測可重現、
     不受攝影機幀率波動影響；即時流程的計算內容與攝影機輸入時完全相同」。

**本程式不複製任何驅動邏輯。** 它 import `gradio_realtime_romp_chupa`，
沿用其已載入的模型、綁定參數與 `process_frame`，僅把輸入由攝影機換成影片。
如此錄到的更新率即為該迴圈的真實表現，不是另一條路徑的數字。

誠實性
======
以影片為輸入所錄到的，**仍是真實的每影格處理時間**——ROMP 推論、姿態轉換、
前向運動學、線性混合蒙皮、glb 匯出全部照跑。與攝影機輸入的差別只在影像的
來源，不在計算量。

惟仍須於影片中標明「輸入為預錄影片」，理由有二：其一，攝影機另有其自身的
擷取延遲與幀率波動，本設定不含該部分；其二，瀏覽器端的影像傳輸路徑不同。
**故本錄影可支持「迴圈的更新率」之宣稱，不足以支持「端到端使用者體驗」
之宣稱。**

用法
====
  conda activate <跑 gradio_realtime_romp_chupa.py 的那個環境>
  cd ~/projects/chupa
  python replay_app.py

  # 指定連接埠
  python replay_app.py --port 7861

開啟 http://localhost:7861 後上傳影片（或直接用 motion_testVideo/1.mp4），
按「開始播放」即逐影格驅動，右側顯示結果、下方顯示實測更新率。

環境：與 gradio_realtime_romp_chupa.py 相同
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="以預錄影片驅動即時迴圈")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--default-video",
                    default="motion_testVideo/1.mp4")
    ap.add_argument("--warmup", type=int, default=20,
                    help="不計入更新率統計的前置影格數。前若干影格含 CUDA "
                         "初始化與核心編譯，耗時可達穩態數倍（4.7.1 節）")
    args = ap.parse_args()

    import cv2
    import gradio as gr

    # 匯入既有程式：其模組層級會載入 SMPL-X、蒙皮權重與 ROMP，
    # 故此處不重複初始化。`demo.launch` 僅在其 __main__ 下執行，匯入安全。
    print("[INFO] 載入 gradio_realtime_romp_chupa（會一併初始化模型）...")
    import gradio_realtime_romp_chupa as rt

    if not hasattr(rt, "process_frame"):
        sys.exit("[錯誤] 找不到 process_frame——該檔的結構可能已變動。")

    def replay(video_path, stride, limit):
        r"""逐影格送入既有的 process_frame，並回報實測更新率。

        **輸入影格與驅動結果由同一次 yield 一併送出**，故二者於畫面上
        逐格對齊，不會因兩個元件各自更新而產生漂移。此為同步展示的必要
        條件——若讓影片自行播放、三維結果另行更新，二者的對應關係即不
        可靠，而觀眾無從察覺。
        """
        if not video_path:
            yield None, None, "請先選擇影片"
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            yield None, None, f"打不開影片：{video_path}"
            return

        n, used, dts, n_miss = 0, 0, [], 0
        t_wall = time.time()
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            n += 1
            if stride > 1 and (n - 1) % stride:
                continue

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()
            out = rt.process_frame(rgb)
            dt = time.time() - t0

            used += 1
            if out is None:
                n_miss += 1
            # 暖機影格不計入統計
            if used > args.warmup:
                dts.append(dt)

            if dts:
                mean = sum(dts) / len(dts)
                msg = (f"影格 {used}（讀入 {n}）　"
                       f"本格 {dt * 1000:.1f} ms　"
                       f"穩態平均 {mean * 1000:.1f} ms"
                       f"（{1.0 / mean:.1f} FPS，已排除前 {args.warmup} 格暖機）")
            else:
                msg = (f"影格 {used}／暖機中（前 {args.warmup} 格不計入統計）　"
                       f"本格 {dt * 1000:.1f} ms")
            if n_miss:
                msg += f"　未偵測到人 {n_miss} 格"

            # 輸入影格與驅動結果同時送出，確保畫面逐格對齊
            yield rgb, out, msg

            if limit and used >= limit:
                break
        cap.release()

        if dts:
            mean = sum(dts) / len(dts)
            srt = sorted(dts)
            p95 = srt[int(0.95 * (len(srt) - 1))]
            wall = time.time() - t_wall
            yield rgb, out, (
                f"**完成**：共處理 {used} 影格（讀入 {n}）。\n\n"
                f"穩態每影格平均 **{mean * 1000:.2f} ms（{1.0 / mean:.1f} FPS）**、"
                f"P95 {p95 * 1000:.2f} ms；總牆鐘時間 {wall:.1f} 秒。\n\n"
                f"未偵測到人之影格：{n_miss}。\n\n"
                f"**輸入為預錄影片。** 此數值涵蓋 ROMP 推論、姿態轉換、"
                f"前向運動學、線性混合蒙皮與 glb 匯出，"
                f"**不含攝影機擷取與瀏覽器端的傳輸延遲**，"
                f"亦不含三維檢視器的繪製。")
        else:
            yield rgb, out, f"處理 {used} 影格，不足以扣除暖機（需 > {args.warmup}）"

    with gr.Blocks() as app:
        gr.Markdown("## 以預錄影片驅動即時迴圈"
                    "\n\n與攝影機輸入**共用同一條處理路徑**；"
                    "差別僅在影像來源，計算內容相同。"
                    "\n\n> 本頁所量得者為**迴圈的更新率**，"
                    "不含攝影機擷取與瀏覽器傳輸的延遲。")

        # 播放時左右並排、逐格同步：左為當下送入的影格，右為其驅動結果。
        with gr.Row():
            live_in = gr.Image(label="輸入影格（當下送入者）")
            out3d = gr.Model3D(label="驅動結果（同一影格）")

        vid = gr.Video(label="輸入影片（選檔用）",
                       value=args.default_video
                       if os.path.exists(args.default_video) else None)

        with gr.Row():
            stride = gr.Slider(1, 5, value=1, step=1, label="每幾格取一")
            limit = gr.Number(value=0, label="最多處理幾格（0 = 全部）",
                              precision=0)
            btn = gr.Button("開始播放", variant="primary")

        status = gr.Markdown()
        btn.click(replay, [vid, stride, limit], [live_in, out3d, status])

        gr.Markdown(
            "### 錄影前的提醒\n"
            "- **先空跑一輪再錄。** 前若干影格含 CUDA 初始化與核心編譯，"
            "其耗時可達穩態數倍（4.7.1 節）；冷啟動的卡頓不代表穩態表現。\n"
            "- **錄影時勿觸發背景生成。** 背景生成期間更新率降至 4.3 FPS、"
            "GPU 記憶體用量達 96.6%（4.7.3 節），該情形不宜混入即時性的展示。\n"
            "- 影片中請標明「輸入為預錄影片」。")

    print(f"[INFO] 啟動於 http://localhost:{args.port}")
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
