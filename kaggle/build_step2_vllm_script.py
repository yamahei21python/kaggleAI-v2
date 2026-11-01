#!/usr/bin/env python3
"""vLLM 配信カーネルを **script 形式**（main.py 1 ファイル）で生成する。

    python3 kaggle/build_step2_vllm_script.py              # cloudflared
    python3 kaggle/build_step2_vllm_script.py --smoke      # 起動＋会話だけ

`STEP2_TUNNEL` は互換のために受けるが cloudflared のみ（ngrok は 2026-11-01 に
retired。`notes/url-publishing.md`）。

notebook 版（`build_step2_vllm.py`）はそのまま残す。こちらは同じセル列
（`kaggle/step2-vllm/cells/*.py`）を `kbuild/serve_script.py` の骨組みに載せた
**追加の生成物**で、notebook を置き換えない。

骨組み（proxy / tunnel / Gist / keepalive / verify / 後片付け）は書かない。
ここが持つのは vLLM 固有の install / launch / checks / warmup だけ。
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import (embed, metadata, pack_script,  # noqa: E402
                     serve_script, spec)
from kbuild.settings_cell import render as render_settings  # noqa: E402
from runtime import proxy_source, source_bundle  # noqa: E402
from runtime.constants import PROXY_PORT, VLLM_PORT  # noqa: E402

CFG = kcconfig.load(ROOT)
OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
SERVED_NAME = "qwen3.8-27b-awq"

TUNNEL = os.environ.get("STEP2_TUNNEL", "cloudflared").strip().lower()
if TUNNEL != "cloudflared":
    raise SystemExit(
        f"STEP2_TUNNEL={TUNNEL!r} は未対応。cloudflared だけ"
        "（ngrok は 2026-11-01 に廃止。notes/url-publishing.md）")

# smoke 版は通常版と別ディレクトリへ書く（`cli push --smoke` が通常版を
# 上書きしないため。notebook 版の `smoke_out` と同じ考え方）。
SMOKE = "--smoke" in sys.argv or os.environ.get("KC_SMOKE") == "1"

CELLS = HERE / "step2-vllm" / "cells"
OUT = HERE / "step2-vllm-script" / (f"smoke-{TUNNEL}" if SMOKE else TUNNEL)

PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(ROOT / "llama-chat.html")

SETTINGS = render_settings(CELLS / "00_settings.py", {
    "__DATASET_SLUG__": DATASET_SLUG,
    "__SERVED__": SERVED_NAME,
    "__TUNNEL__": TUNNEL,
    "__PROXY_B64__": PROXY_SRC_B64,
    "__CHAT_B64__": CHAT_SRC_B64,
    "__GIST_ID__": CFG.get("GIST_ID", ""),
    "__GIST_OWNER__": CFG.get("GIST_OWNER", ""),
})

ENGINE_CODE = (
    pack_script.wrap_cell(CELLS / "01_install_vllm.py", "def stage_install():\n")
    + "\n\n"
    + pack_script.wrap_cell(
        CELLS / "03_vllm_serve.py", "def launch(model_dir):\n",
        # 起動セルは実測した max_model_len で CONTEXT_LENGTH を上書きする。
        # 関数化するとローカル変数になるため、モジュール側を更新させる。
        tail="\n    return proc\n",
        globals_=("CONTEXT_LENGTH",))
)

ENGINE = {
    "name": "vllm",
    "version": "0.32.0",
    "alias": SERVED_NAME,
    "pattern": "-f vllm",
    "settings": SETTINGS,
    "engine_code": ENGINE_CODE,
    "checks_code": (CELLS / "06_checks.py").read_text(encoding="utf-8"),
    "warmup": pack_script.wrap_cell(CELLS / "04_warmup.py", "def warmup():\n"),
    "work": "/tmp/vllm-serve",
    "flow": {
        # モデルは設定セルの MODEL_SRC（Dataset 直下）を使う。
        # find_model() の config.json 探索は使わない。
        "model": False,
        "tp": True,
        "install": True,
        "publish": True,
        "verify_public": True,
        "keepalive": False,
        "conf": True,
    },
    "data_init": ('{"engine": ENGINE, "stage": "init", "tunnel": TUNNEL_KIND, '
                  '"model": MODEL_NAME, "context_length": CONTEXT_LENGTH, '
                  '"smoke": SMOKE}'),
}

SLUG = f"step2-vllm-qwen38-27b-{TUNNEL}-script"
# metadata は台帳から引く。datasets（モデル + cloudflared + wheelhouse 2 点）も
# 台帳側に移した。wheelhouse の patch が要る理由は `kbuild/spec.py` の当該行に
# 書いてある（cuda-toolkit 13.0.3 が `nvidia-cublas` を固定するため）。
META = metadata.for_kernel(spec.find(SLUG))

if __name__ == "__main__":
    serve_script.write(
        OUT,
        ENGINE,
        META,
        run_minutes=60,
        runtime=source_bundle("net", "proc", "storage", "keepalive", "proxy",
                              "gist"),
        proxy_b64=PROXY_SRC_B64,
        html_b64=CHAT_SRC_B64,
        gist_id=CFG.get("GIST_ID", ""),
        gist_owner=CFG.get("GIST_OWNER", ""),
        port=VLLM_PORT,
        proxy_port=PROXY_PORT,
        hard_max_minutes=180,
        smoke=SMOKE,
        tunnel=TUNNEL,
        builder="kaggle/build_step2_vllm_script.py",
    )
