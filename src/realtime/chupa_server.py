"""
chupa_server.py
在 chupa 環境執行的常駐生成服務。
啟動時載入一次所有 diffusion 模型並常駐記憶體，
之後每個 HTTP 請求只做推論。

啟動：
    conda activate chupa
    cd ~/projects/chupa
    python chupa_server.py

測試：
    curl http://localhost:7861/health
    curl -X POST http://localhost:7861/generate \
         -H "Content-Type: application/json" \
         -d '{"prompt": "a woman wearing a long dress"}'
"""

import sys
import json
import time
import importlib.util
import traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

from omegaconf import OmegaConf


# ============================================================
# 用 importlib 從檔案路徑載入 scripts/chupa.py 的 Chupa 類別
# ============================================================

PROJECT_DIR = Path(__file__).parent.resolve()

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

_chupa_py_path = PROJECT_DIR / "scripts" / "chupa.py"
_spec = importlib.util.spec_from_file_location("chupa_module", str(_chupa_py_path))
_chupa_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chupa_module)
Chupa = _chupa_module.Chupa


# ============================================================
# 啟動時：載入 config、建立 Chupa 物件（模型只載入這一次）
# ============================================================

CONFIG_FILE = "configs/text.yaml"
SUBJECT = "rp_emily_apose_001"
PORT = 7861

print("[SERVER] 載入 config 並建立 Chupa 物件（載入所有模型，需要一點時間）...")

config = OmegaConf.load(CONFIG_FILE)
override = OmegaConf.create({
    "dataset": {"subject": SUBJECT},
    "chupa": {"n_resample_from_text": 0},
    "chupa": {
        "n_resample_from_text": 0,
        "n_resample": 0,
    },
    "nds": {
        "iterations": 1000,
    },
})
config = OmegaConf.merge(config, override)

chupa = Chupa(config)

data_dir = Path(config.dataset.data_dir) / config.dataset.dataset_type
smpl_type = config.chupa.smpl_type
smpl_param_path = data_dir / smpl_type / f"{SUBJECT}.pkl"

print(f"[SERVER] 模型載入完成，subject={SUBJECT}")
print(f"[SERVER] SMPL 參數路徑: {smpl_param_path}")


# ============================================================
# HTTP 請求處理
# ============================================================

class GenerateHandler(BaseHTTPRequestHandler):

    def _send_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            prompt = req.get("prompt", "").strip()

            if not prompt:
                self._send_json(400, {"error": "empty prompt"})
                return

            print(f"\n[SERVER] 收到生成請求: '{prompt}'")
            t0 = time.time()

            chupa.prompt = prompt
            output_mesh_path = chupa.forward(
                smpl_param_path=smpl_param_path,
                subject=SUBJECT,
                prompt=prompt,
            )

            elapsed = time.time() - t0
            print(f"[SERVER] 生成完成，耗時 {elapsed:.1f}s")
            print(f"[SERVER] mesh: {output_mesh_path}")

            self._send_json(200, {
                "mesh_path": str(output_mesh_path),
                "elapsed_sec": elapsed,
            })

        except Exception as e:
            print("[SERVER] 生成失敗:")
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "subject": SUBJECT})
        else:
            self._send_json(404, {"error": "unknown endpoint"})

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), GenerateHandler)
    print(f"[SERVER] 常駐生成服務已啟動，監聽 http://0.0.0.0:{PORT}")
    print("[SERVER] 等待生成請求中...（按 Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] 收到中斷，關閉服務")
        server.server_close()