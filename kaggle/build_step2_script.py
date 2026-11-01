#!/usr/bin/env python3
"""Ollama 配信カーネルを **script 形式**（main.py 1 ファイル）で生成する。

    python3 kaggle/build_step2_script.py            # 通常
    python3 kaggle/build_step2_script.py --smoke    # 疎通だけ（GPU 時間の節約）

notebook 版（`build_step2.py`）はそのまま残す。こちらは同じセル列
（`kaggle/step2/cells/*.py`）を `kbuild/serve_script.py` の骨組みに載せた
**追加の生成物**で、notebook を置き換えない。

骨組みが持つもの（proxy / tunnel / Gist / keepalive / verify / 後片付け）は
一切書かない。ここが持つのは Ollama 固有の 3 つだけ:

    settings / stage_install / launch   （+ 確認項目とウォームアップ）

セルは「トップレベルで書かれた逐次コード」なので、関数へは**機械的に包む**。
包むだけで意味が変わらないため、notebook 版との挙動差が出にくい。
"""
import os
import pathlib
import sys
import textwrap

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent                                      # リポジトリ直下
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import embed, metadata, serve_script, spec  # noqa: E402
from kbuild.settings_cell import render as render_settings  # noqa: E402
from runtime import proxy_source, source_bundle  # noqa: E402
from runtime.constants import PROXY_PORT  # noqa: E402

CFG = kcconfig.load(ROOT)
OWNER = "yamahei21python"

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
MODEL_DISPLAY = "Qwen3.8-27B-Uncensored"
DATASET = "my-ollama-orcarouter-qwen38-27b"
PATCH = "my-ollama-orcarouter-qwen38-27b-patch"
OLLAMA_PORT = 11434

TUNNEL = os.environ.get("STEP2_TUNNEL", "cloudflared").strip().lower()
if TUNNEL != "cloudflared":
    raise SystemExit(
        f"STEP2_TUNNEL={TUNNEL!r} は未対応。cloudflared だけ"
        "（ngrok は 2026-11-01 に廃止。notes/url-publishing.md）")

# smoke 版は通常版と別ディレクトリへ書く（同じだと `cli push --smoke` が
# 通常版を上書きしてしまう）。notebook 版の `smoke_out` と同じ考え方。
SMOKE = "--smoke" in sys.argv or os.environ.get("KC_SMOKE") == "1"

CELLS = HERE / "step2" / "cells"

# 出力先: notebook 版（kaggle/step2/…ipynb）とは別ディレクトリ。
# 同じディレクトリに置くと kernel-metadata.json の code_file が衝突する。
OUT = HERE / "step2-script" / (f"smoke-{TUNNEL}" if SMOKE else TUNNEL)


def cell(name: str) -> str:
    return (CELLS / name).read_text(encoding="utf-8")


def _render_settings() -> str:
    """`00_settings.py` の __TOKEN__ を実値に置換する（notebook 版と共通処理）。"""
    return render_settings(CELLS / "00_settings.py", {
        "__MODEL_DISPLAY__": MODEL_DISPLAY,
        "__MODEL__": MODEL,
        "__DATASET__": DATASET,
        "__PATCH__": PATCH,
        "__TUNNEL__": TUNNEL,
        "__PROXY_B64__": PROXY_SRC_B64,
        "__CHAT_B64__": CHAT_SRC_B64,
        "__GIST_ID__": CFG.get("GIST_ID", ""),
        "__GIST_OWNER__": CFG.get("GIST_OWNER", ""),
    })


def wrap(name: str, header: str, tail: str = "") -> str:
    """セル（トップレベルコード）を関数本体へ機械的に包む。

    セルを書き換えず、インデントを付けるだけ。`tail` は戻り値の確定に使う
    （起動セルは `proc` を返す必要がある）。
    """
    body = cell(name).rstrip()
    return header + textwrap.indent(body, "    ") + tail


PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(ROOT / "llama-chat.html")

ENGINE_CODE = (
    wrap("01_restore_binaries.py", "def stage_install():\n") + "\n\n"
    + wrap("03_ollama_serve.py", "def launch(model_dir):\n",
           "\n    return proc\n")
)

ENGINE = {
    "name": "ollama",
    "version": "",
    # Ollama のモデル名は `repo:tag` 形式。ALIAS は OpenAI 互換の model 名。
    "alias": MODEL,
    "pattern": "-x ollama",
    "settings": _render_settings(),
    "engine_code": ENGINE_CODE,
    "checks_code": cell("06_checks.py"),
    "warmup": wrap("04_warmup.py", "def warmup():\n"),
    # 設定セルが engine_cleanup() を持つ。骨組みの cleanup() から呼ぶ。
    "cleanup": "cleanup = engine_cleanup\n",
    "work": "/tmp/ollama-serve",
    "flow": {
        # モデルは Dataset の blob を OLLAMA_MODELS へ展開する方式なので
        # find_model()（config.json 探索）は使わない。
        "model": False,
        "tp": False,
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

SLUG = f"step2-orcarouter-qwen38-27b-{TUNNEL}-script"
# metadata は台帳から引く（datasets も台帳側）。
META = metadata.for_kernel(spec.find(SLUG))

if __name__ == "__main__":
    serve_script.write(
        OUT,
        ENGINE,
        META,
        run_minutes=60,
        runtime=source_bundle("net", "proc", "storage", "keepalive",
                              "proxy", "ollama"),
        proxy_b64=PROXY_SRC_B64,
        html_b64=CHAT_SRC_B64,
        gist_id=CFG.get("GIST_ID", ""),
        gist_owner=CFG.get("GIST_OWNER", ""),
        port=OLLAMA_PORT,
        proxy_port=PROXY_PORT,
        hard_max_minutes=480,
        smoke=SMOKE,
        tunnel=TUNNEL,
        builder="kaggle/build_step2_script.py",
    )
