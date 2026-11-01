#!/usr/bin/env python3
"""llama-server 直起動の配信カーネルを **script 形式**（main.py 1 ファイル）で生成する。

    python3 kaggle/build_llama_server_script.py

notebook 版（`build_llama_server.py`）はそのまま残す。こちらは同じセル列
（`kaggle/llama-server/cells/*.py`）を `kbuild/serve_script.py` の骨組みに載せた
**追加の生成物**で、notebook を置き換えない。

設定セルは notebook 版のビルダーから借りる（`build_llama_server.SETTINGS`）。
同じ定義を 2 箇所に書かないため、片方だけ古いという事故が起きない。
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TUNNEL = os.environ.get("LLAMA_TUNNEL", "ngrok").strip().lower()
if TUNNEL not in ("cloudflared", "ngrok"):
    raise SystemExit(f"LLAMA_TUNNEL={TUNNEL!r} は未対応（cloudflared / ngrok）")
# 借用する前に環境変数を確定させる（import 時に読まれる）
os.environ["LLAMA_TUNNEL"] = TUNNEL

sys.path.insert(0, str(HERE))
import build_llama_server as base  # noqa: E402
from kbuild import metadata, pack_script, serve_script  # noqa: E402
from runtime import source_bundle  # noqa: E402
from runtime.constants import PROXY_PORT  # noqa: E402

CELLS = base.CELLS_DIR
LLAMA_PORT = 11434
OWNER = "yamahei21python"

# smoke 版は通常版と別ディレクトリへ書く（`cli push --smoke` が通常版を
# 上書きしないため。notebook 版の `smoke_out` と同じ考え方）。
SMOKE = "--smoke" in sys.argv or os.environ.get("KC_SMOKE") == "1"

OUT = HERE / "llama-server-script" / (f"smoke-{TUNNEL}" if SMOKE else TUNNEL)

ENGINE_CODE = (
    pack_script.wrap_cell(CELLS / "01_restore_llama.py",
                          "def stage_install():\n")
    + "\n\n"
    + pack_script.wrap_cell(CELLS / "03_llama_serve.py",
                            "def launch(model_dir):\n",
                            tail="\n    return proc\n")
)

ENGINE = {
    "name": "llama-server",
    "version": "",
    "alias": base.MODEL,
    "pattern": "-f llama-server",
    "settings": base.SETTINGS,
    "engine_code": ENGINE_CODE,
    "checks_code": (CELLS / "06_checks.py").read_text(encoding="utf-8"),
    "work": "/tmp/llama-server-serve",
    "flow": {
        # モデルは設定セルの DATASET_PATH（GGUF）から解決する。
        "model": False,
        "tp": False,
        "install": True,
        "publish": True,
        "verify_public": True,
        "keepalive": False,
        "conf": True,
    },
    "data_init": ('{"engine": ENGINE, "stage": "init", "tunnel": TUNNEL_KIND, '
                  '"model": MODEL_NAME, "smoke": SMOKE}'),
}

SLUG = f"llama-server-{TUNNEL}-script"
META = metadata.base(
    id=f"{OWNER}/{SLUG}",
    # エンジン名のハイフンは残す（slugify しても slug は変わらない）。
    title=f"llama-server {TUNNEL} script",
    code_file="main.py",
    kernel_type="script",
    gpu=True,
    datasets=[f"{OWNER}/{base.DATASET}", f"{OWNER}/{base.PATCH}"],
)

if __name__ == "__main__":
    serve_script.write(
        OUT,
        ENGINE,
        META,
        run_minutes=60,
        runtime=source_bundle("net", "proc", "storage", "keepalive", "proxy"),
        proxy_b64=base.PROXY_SRC_B64,
        html_b64=base.CHAT_SRC_B64,
        gist_id=base.CFG.get("GIST_ID", ""),
        gist_owner=base.CFG.get("GIST_OWNER", ""),
        port=LLAMA_PORT,
        proxy_port=PROXY_PORT,
        hard_max_minutes=240,
        smoke=SMOKE,
        tunnel=TUNNEL,
        ngrok_domain=base.CFG.get("NGROK_DOMAIN", ""),
        builder="kaggle/build_llama_server_script.py",
    )
