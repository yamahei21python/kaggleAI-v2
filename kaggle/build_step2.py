#!/usr/bin/env python3
"""Step2 notebook builder: orcarouter/Qwen3.8-27B-Uncensored + Ollama on Kaggle T4 x2

セルの並び・配管（proxy / tunnel / Gist / verify / 接続情報 / keepalive）は
`kbuild/serve_notebook.py` が唯一の定義。ここが持つのは **Ollama 固有の 4 つ**だけ:

    settings / install / launch / verify_checks

トンネルは環境変数で切り替える（セル構成は共通、トンネルセルだけ差し替え）:
    STEP2_TUNNEL=cloudflared  (既定) → アカウント不要・URL は毎回ランダム
    STEP2_TUNNEL=ngrok                → NGROK_TOKEN 必須・NGROK_DOMAIN で固定URL可
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import kcconfig  # noqa: E402
from kbuild import embed, metadata  # noqa: E402
from kbuild.settings_cell import render as render_settings  # noqa: E402
from kbuild.serve_notebook import Flow  # noqa: E402
from runtime import proxy_source  # noqa: E402

CFG = kcconfig.load()

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
MODEL_DISPLAY = "Qwen3.8-27B-Uncensored"
DATASET = "my-ollama-orcarouter-qwen38-27b"
PATCH = "my-ollama-orcarouter-qwen38-27b-patch"

TUNNEL = os.environ.get("STEP2_TUNNEL", "cloudflared").strip().lower()
if TUNNEL not in ("cloudflared", "ngrok"):
    raise SystemExit(f"STEP2_TUNNEL={TUNNEL!r} は未対応（cloudflared / ngrok）")
IS_NGROK = TUNNEL == "ngrok"

# 出力先・プロキシ置き場は __file__ 基準。cwd に依存させない。
# （相対パスだとリポジトリ直下で走らせたときに別所へ生成される）
HERE = pathlib.Path(__file__).resolve().parent
OUT = str(HERE / ("step2-ngrok" if IS_NGROK else "step2") /
          f"step2-orcarouter-qwen38-27b-{TUNNEL}.ipynb")

# Basic 認証つきリバースプロキシ / チャット UI を notebook に埋め込む（base64）。
# 単一のソースをそのまま使うので、notebook 側でコードを二重管理しない。
PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(HERE.parent / "llama-chat.html")

CELLS_DIR = HERE / "step2" / "cells"


def _cell(name):
    return (CELLS_DIR / name).read_text(encoding="utf-8")


# ── title ───────────────────────────────────────────────────
cf_bullet = ("" if IS_NGROK else
             "   - `my-cloudflared-bin`（cloudflared 40 MB）"
             "— 無いと毎回 GitHub から落とす")
secret_note = (
    """> ✅ **必要な Secrets**: `NGROK_TOKEN` のみ（必須）
> `NGROK_DOMAIN` はノートに埋め込み済み → **毎回同じ URL**。
> Secret に入れると上書きできる（未設定なら埋め込み値を使う）。
>
> 🔓 **認証はありません**（URL を知れば誰でも使えます）。
> 他人に使われたくないときは環境変数 `API_PASS` を設定してください
> （proxy が Basic 認証を有効化します）。"""
    if IS_NGROK else
    """> ✅ **Secrets の設定は不要です。**
> cloudflared の quick tunnel はアカウント不要。authtoken も API キーも要りません。
> 自動で発行される公開URLがそのままエンドポイントになります。
>
> 🔓 **認証はありません**（URL を知れば誰でも使えます）。
> 他人に使われたくないときは環境変数 `API_PASS` を設定してください
> （proxy が Basic 認証を有効化します）。
>
> ⚠ 公開URLは **実行のたびに変わります**（quick tunnel の仕様）。
> 切れたらトンネルセルを再実行し、接続情報を取り直してください。""")

TITLE = f"""# 🚀 Step 2: Ollama + {TUNNEL} 公開APIサーバー

**{MODEL}** を Kaggle の T4 x2 で動かし、{TUNNEL} で公開します。

---

### 前提条件（これだけ）

1. **Accelerator**: `GPU T4 x2` に設定
2. **Internet**: `On` に設定
3. **Add Data**: 以下の **{2 if IS_NGROK else 3}つ** を追加
   - `{DATASET}`（モデル 16.5 GiB）
   - `{PATCH}`（ollama + llama-server + CUDA 1.45 GB）
{cf_bullet}
{secret_note}

---

### このモデルについて

| 項目 | 値 |
|---|---|
| 量子化 | Q4_K_M |
| パラメータ | 27.3B |
| サイズ | 16.52 GiB |
| コンテキスト | 262,144（このノートブックでは 131,072 で運用） |
| 機能 | completion / vision / tools / thinking |

Abliterlitics の Qwen3.8-27B 12変体比較で総合1位（judge ASR 82.2%、capability 劣化なし）。
手元の `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` とは sha256 完全一致。
"""

# ── settings（Ollama 固有）────────────────────────────────────
# 定義は kaggle/step2/cells/00_settings.py が唯一の出所。
# script 版（build_step2_script.py）も同じファイルを読むので、
# 片方だけ直して片方が古い、という事故が起きない。
SETTINGS = render_settings(CELLS_DIR / "00_settings.py", {
    "__MODEL_DISPLAY__": MODEL_DISPLAY,
    "__MODEL__": MODEL,
    "__DATASET__": DATASET,
    "__PATCH__": PATCH,
    "__TUNNEL__": TUNNEL,
    "__PROXY_B64__": PROXY_SRC_B64,
    "__CHAT_B64__": CHAT_SRC_B64,
    "__GIST_ID__": CFG["GIST_ID"],
    "__GIST_OWNER__": CFG["GIST_OWNER"],
    "__NGROK_DOMAIN__": CFG["NGROK_DOMAIN"],
})

flow = Flow(
    title=TITLE,
    settings=SETTINGS,
    install=_cell("01_restore_binaries.py"),
    launch=_cell("03_ollama_serve.py"),
    verify_checks=_cell("06_checks.py"),
    warmup=_cell("04_warmup.py"),
    tunnel=TUNNEL,
    install_md="---\n### 📡 Ollama / cloudflared インストール（Dataset キャッシュ優先）\n",
    launch_md="---\n### 🔥 Ollama サーバー起動\n",
    verify_md="---\n### ✅ 公開URL経由の疎通確認（UI / chat / streaming / OpenAI互換 / GPU常駐）\n",
    # runtime/engines/ollama.py の kc_unload_model / kc_wait_vram_free を使う
    runtime=("net", "proc", "storage", "keepalive", "proxy", "ollama"),
)
flow.write(OUT)

# ── kernel-metadata.json（push 用）────────────────────────────
# トンネルごとに別 kernel として push できるよう、id と sources を切り替える。
SLUG = ("step2-orcarouter-qwen38-27b-ngrok-fixed" if IS_NGROK
        else "step2-orcarouter-qwen38-27b-cloudflared")
sources = [f"yamahei21python/{DATASET}", f"yamahei21python/{PATCH}"]
if not IS_NGROK:
    sources.append("yamahei21python/my-cloudflared-bin")
meta = {
    "id": f"yamahei21python/{SLUG}",
    # title は slug に解決できる必要がある（不一致だと push が 409 Conflict）
    "title": (SLUG.replace("-", " ") if IS_NGROK
              else "Step2 orcarouter qwen38 27b cloudflared"),
    "code_file": pathlib.Path(OUT).name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": sources,
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
meta_path = pathlib.Path(OUT).with_name("kernel-metadata.json")
metadata.write(meta_path, meta)
print(f"wrote {meta_path}")
