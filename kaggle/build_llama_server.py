#!/usr/bin/env python3
"""llama-server 直起動版ビルダー: llama-server を Ollama 経由せず直接起動する。

セルの並び・配管は `kbuild/serve_notebook.py` が唯一の定義。ここが持つのは
**llama-server 固有の 4 つ**（settings / install / launch / verify_checks）だけ。

なぜ直起動か:
  Ollama 0.40.12 は MTP（投機的デコード）のフラグを一切露出していない。
  `ollama serve --help` の環境変数 19 個に spec / mtp / draft が無い。
  llama-server を直に叩けば --spec-type draft-mtp が使える。

代償:
  Ollama が担っていた層を自前で持つ。ただし Dataset に model blob と
  projector blob が既にあるので、pull は不要。

環境変数:
  LLAMA_TUNNEL=cloudflared         既定 cloudflared（ngrok は 2026-11-01 に廃止）
  LLAMA_MTP=1|0                    既定 1（MTP 有効）
  LLAMA_CTX=<int>                  既定 32768（テスト用に小さめ）
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import embed, metadata  # noqa: E402
from kbuild.serve_notebook import Flow  # noqa: E402
from runtime import proxy_source  # noqa: E402

CFG = kcconfig.load()

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
MODEL_DISPLAY = "Qwen3.8-27B-Uncensored (直起動)"
DATASET = "my-ollama-orcarouter-qwen38-27b"
PATCH = "my-ollama-orcarouter-qwen38-27b-patch"

# ngrok は 2026-11-01 に廃止した（`notes/url-publishing.md`）。llama-server の
# 配信カーネルは ngrok 前提で作っていたので台帳では retired にしてあるが、
# ビルダー自体は cloudflared で生成できる状態に残す（作り直したくなったとき用）。
TUNNEL = os.environ.get("LLAMA_TUNNEL", "cloudflared").strip().lower()
if TUNNEL != "cloudflared":
    raise SystemExit(
        f"LLAMA_TUNNEL={TUNNEL!r} は未対応。cloudflared だけ"
        "（ngrok は 2026-11-01 に廃止）")

USE_MTP = os.environ.get("LLAMA_MTP", "1").strip() == "1"
CTX = int(os.environ.get("LLAMA_CTX", "32768"))

# 出力はセル置き場も兼ねる `llama-server/`（`kaggle/step2/` と同じ関係）。
OUT_DIR = HERE / "llama-server"
OUT = OUT_DIR / f"llama-server-{TUNNEL}.ipynb"

PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(ROOT / "llama-chat.html")

CELLS_DIR = HERE / "llama-server" / "cells"


def _cell(name):
    return (CELLS_DIR / name).read_text(encoding="utf-8")


# ── title ───────────────────────────────────────────────────
TITLE = f"""# 🚀 llama-server + {TUNNEL}

**{MODEL}** を Ollama を経由せず **llama-server 直起動**で動かします。

---

### なぜ直起動か

Ollama 0.40.12 は **MTP（投機的デコード）のフラグを露出していません**。
`ollama serve --help` の環境変数 19 個に `spec` / `mtp` / `draft` が 1 つも無いため、
Modelfile に `PARAMETER spec_type draft_mtp` と書いても効きません。

llama-server を直に叩けば `--spec-type draft-mtp` が使えます。

| | Ollama 経由（既存） | 直起動（このノート） |
|---|---|---|
| 起動 | `ollama serve` | `llama-server --model ...` |
| MTP | ❌ 露出なし | ✅ `--spec-type draft-mtp` |
| フラグ制御 | ❌ | ✅ 全部 |
| モデル管理 | ✅ pull / Modelfile | Dataset の blob を直接指定 |

### 前提条件

1. **Accelerator**: `GPU T4 x2`
2. **Internet**: `On`
3. **Add Data**: `{DATASET}` と `{PATCH}`

> **Secrets は不要です**（cloudflared はアカウント不要）。

### このノートの設定

| 項目 | 値 |
|---|---|
| MTP | {'有効（--spec-type draft-mtp）' if USE_MTP else '無効'} |
| コンテキスト | {CTX:,} |
| KV キャッシュ | q8_0 |
| thinking | off |

> ⚠ **テスト用にコンテキストを {CTX:,} に抑えています。** 本番は 131,072。
"""

# ── settings（llama-server 固有）──────────────────────────────
SETTINGS = f'''# ============================================================
# Step 2 (直起動): llama-server を Ollama 経由せず起動
# ============================================================
import os

MODEL_NAME = "{MODEL}"
MODEL_DISPLAY = "{MODEL_DISPLAY}"
ENGINE_DISPLAY = "llama-server"
ENGINE_LABEL = "llama-server"

DATASET_SLUG = "{DATASET}"
PATCH_SLUG = "{PATCH}"
USE_MTP = {USE_MTP}
CTX_REQUESTED = {CTX}

# 直起動では Ollama のポートと別にする（既存ノートと衝突させない）
PORT = UPSTREAM_PORT = 11434
PROXY_PORT = 11435
PROXY_CHECK_PATH = "/api/tags"
PROXY_EXTRA_ARGS = ()
TUNNEL_KIND = "{TUNNEL}"

PROXY_SRC_B64 = "{PROXY_SRC_B64}"
CHAT_SRC_B64 = "{CHAT_SRC_B64}"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

LIB_DIR = "/usr/local/lib/ollama"
LLAMA_SERVER = os.path.join(LIB_DIR, "llama-server")

# モデルデータの配置先。Ollama 版と同じステージング処理を使うため同じ変名。
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"

# Dataset のマウント先は共通ランタイム（runtime/storage.py）に一本化。
DATASET_PATH = kc_find_dataset(DATASET_SLUG)
PATCH_PATH = kc_find_dataset(PATCH_SLUG)

# 認証は既定で無効。API_PASS が入っているときだけ proxy が有効化する。
PROXY_AUTH = {{
    "API_USER": os.environ.get("API_USER", "") or "ollama",
    "API_PASS": os.environ.get("API_PASS", ""),
}}

print(f"DATASET : {{DATASET_PATH}}")
print(f"PATCH   : {{PATCH_PATH}}")

_gpu_names, _vram_each = kc_detect_gpu()
VRAM_TOTAL = sum(_vram_each)
print(f"GPU      : {{_gpu_names}}")
print(f"VRAM     : {{_vram_each}} MiB each  →  合計 {{VRAM_TOTAL}} MiB")

if VRAM_TOTAL == 0:
    raise RuntimeError("GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")

# ---- コンテキスト長 ----
# 実測（Ollama 経由・T4 x2）: 32K=13.7 tps / 131K=13.7 tps / 262K=3.7 tps
# 直起動でも同じモデル・同じ量子化なので近似できる。
MODEL_GIB = 16.52
KV_KIB_PER_TOK_Q8 = 34.0

# 予算（実効 VRAM の 88%）と候補の当てはめは runtime/proc.py に一本化。
# CTX_REQUESTED（環境変数）が収まるならそれを優先する。
CONTEXT_LENGTH, budget_gib = kc_context_budget(
    VRAM_TOTAL, MODEL_GIB, KV_KIB_PER_TOK_Q8,
    [131072, 65536, 32768, 16384, 8192], requested=CTX_REQUESTED)
print(f"予算     : {{budget_gib:.1f}} GiB（実効 VRAM の 88%）")
if CONTEXT_LENGTH != CTX_REQUESTED:
    print(f"⚠ {{CTX_REQUESTED:,}} は予算超過 → {{CONTEXT_LENGTH:,}} に下げる")

kv_used = KV_KIB_PER_TOK_Q8 * CONTEXT_LENGTH / 1024 / 1024
print(f"コンテキスト: {{CONTEXT_LENGTH:,}} tokens")
print(f"  モデル {{MODEL_GIB}} GiB + KV(q8_0) {{kv_used:.2f}} GiB = {{MODEL_GIB + kv_used:.2f}} GiB")
print(f"MTP        : {{'有効' if USE_MTP else '無効'}}")

# ---- Secrets の既定値（config.env 由来。Secret があれば上書きされる）----
GIST_ID_DEFAULT = "{CFG["GIST_ID"]}"
GIST_OWNER = "{CFG["GIST_OWNER"]}"

# ---- キープアライブ ----
MAX_MINUTES = 480
KEEPALIVE_RESET_MINUTES = 60
TUNNEL_CHECK_EVERY = 10

# ---- 死活監視 / 後片付けの能力表（共有セルが読む）----
ENGINE_LOG = "/tmp/llama-server.log"
ENGINE_PROCS = [("llama-server", "argv"), ("auth_proxy.py", "argv")]
CLEANUP_TARGETS = [
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f llama-server", "llama-server"),
]


def engine_cleanup():
    """llama-server にはアンロード API が無い（プロセスを落とすだけ）。"""
    return None


# ---- 検証成果物の永続化 ----
OUT_DIRS = KC_OUT_DIRS
save_output = kc_save_output
fail = kc_fail
'''

if __name__ == "__main__":
    flow = Flow(
        title=TITLE,
        settings=SETTINGS,
        install=_cell("01_restore_llama.py"),
        launch=_cell("03_llama_serve.py"),
        verify_checks=_cell("06_checks.py"),
        tunnel=TUNNEL,
        install_md="---\n### 📡 llama-server + CUDA ライブラリ復元（Dataset から）\n",
        launch_md="---\n### 🔥 llama-server 起動（Ollama を経由しない）\n",
        verify_md="---\n### ✅ 公開URL経由の疎通確認 + 速度測定（MTP の効果）\n",
        keepalive_md=(
            "---\n### 🔋 キープアライブ\n\n"
            "有限ループで自動終了します（`KEEPALIVE_RESET_MINUTES` / `MAX_MINUTES`）。\n\n"
            "- チャット送信で残り時間がリセットされます（`/tmp/keepalive.conf`）\n"
            "- 「■ 停止」ボタンで `RUN=0` が書かれ、ループが終了します\n"),
    )
    flow.write(OUT)

    # ── kernel-metadata.json ────────────────────────────────────
    meta = {
        "id": f"yamahei21python/llama-server-{TUNNEL}",
        "title": f"llama-server {TUNNEL}",
        "code_file": OUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            f"yamahei21python/{DATASET}",
            f"yamahei21python/{PATCH}",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    metadata.write(OUT_DIR / "kernel-metadata.json", meta)
    print(f"wrote {OUT_DIR / 'kernel-metadata.json'}")
