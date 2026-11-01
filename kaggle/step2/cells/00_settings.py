# ============================================================
# Step 2: Ollama 配信カーネルの設定
# ============================================================
# ビルダーが __TOKEN__ を置換して埋め込む。notebook でも script でも
# この 1 ファイルが唯一の定義（両方の生成物が同じ定数を読む）。
import os

MODEL_NAME = "__MODEL__"
MODEL_DISPLAY = "__MODEL_DISPLAY__"
ENGINE_DISPLAY = "Ollama"
ENGINE_LABEL = "ollama"

# Dataset のマウント先は共通ランタイム（runtime/storage.py）に一本化。
# 旧 /kaggle/input/<slug> と新 /kaggle/input/datasets/<owner>/<slug> の
# 両対応・空マウント回避は kc_find_dataset が持つ（ビルダー側で書かない）。
DATASET_PATH = kc_find_dataset("__DATASET__")
PATCH_PATH = kc_find_dataset("__PATCH__")
# cloudflared 専用 Dataset。フォルダ構造は flatten されることがあるので
# インストール側で bin/ の有無両方を探索する。
CF_PATH = kc_find_dataset("my-cloudflared-bin")
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
OLLAMA_BIN = "/usr/local/bin/ollama"

# ---- ポート ----
# Ollama 本体は 11434 のまま、公開側はプロキシが 11435 で受ける。
PORT = UPSTREAM_PORT = 11434
PROXY_PORT = 11435
PROXY_CHECK_PATH = "/api/tags"
PROXY_EXTRA_ARGS = ()
# トンネル種別（ビルダーが埋める）。インストール側が
# 「cloudflared を復元するか」を分岐するために使う。
TUNNEL_KIND = "__TUNNEL__"

# Basic 認証プロキシのソース（base64）。ビルダーが auth_proxy.py を埋め込む。
PROXY_SRC_B64 = "__PROXY_B64__"
# チャット UI（proxy が / で返す）
CHAT_SRC_B64 = "__CHAT_B64__"
CHAT_PATH = "/usr/local/bin/llama-chat.html"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"

# 認証は既定で無効。API_PASS が入っているときだけ proxy が有効化する。
# cloudflared の quick tunnel に Basic 認証は無いので、UI を開いただけで
# 使えることを優先している（URL を知れば誰でも使える点は許容）。
PROXY_AUTH = {
    "API_USER": os.environ.get("API_USER", "") or "ollama",
    "API_PASS": os.environ.get("API_PASS", ""),
}


# ---- GPU / コンテキスト長 ----
_gpu_names, _vram_each = kc_detect_gpu()
VRAM_TOTAL = sum(_vram_each)          # ← 合計で判定する（per-GPU ではない）
print(f"GPU      : {_gpu_names}")
print(f"VRAM     : {_vram_each} MiB each  →  合計 {VRAM_TOTAL} MiB")

if VRAM_TOTAL == 0:
    raise RuntimeError("GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")

# 実測: full attention 17層 × kv_head 4 × (key 256 + value 256) × 2B = 68 KiB/token (f16)
#       q8_0 KV ならその半分 = 34 KiB/token
MODEL_GIB = 16.52
KV_KIB_PER_TOK_Q8 = 34.0

# 実測（kaggle/bench/）: 32K=13.7 tps / 131K=13.7 tps / 262K=3.7 tps。
# 131K までは速度が落ちない。262K で 1/3.7 になるので既定は 131K。
CTX_CANDIDATES = [131072, 65536, 32768, 16384, 8192]
# 予算（実効 VRAM の 88%）と候補の当てはめは runtime/proc.py に一本化。
CONTEXT_LENGTH, budget_gib = kc_context_budget(
    VRAM_TOTAL, MODEL_GIB, KV_KIB_PER_TOK_Q8, CTX_CANDIDATES)
print(f"予算     : {budget_gib:.1f} GiB（実効 VRAM の 88%）")

kv_used = KV_KIB_PER_TOK_Q8 * CONTEXT_LENGTH / 1024 / 1024
print(f"コンテキスト: {CONTEXT_LENGTH:,} tokens")
print(f"  モデル {MODEL_GIB} GiB + KV(q8_0) {kv_used:.2f} GiB = {MODEL_GIB + kv_used:.2f} GiB")

# ---- Secrets の既定値（config.env 由来。Secret があれば上書きされる）----
GIST_ID_DEFAULT = "__GIST_ID__"
GIST_OWNER = "__GIST_OWNER__"

# ---- キープアライブ ----
MAX_MINUTES = 480                 # セル開始からの合計上限（Kaggle 9h より短く）
KEEPALIVE_RESET_MINUTES = 60      # チャット送信ごとにリセットする時間（分）
TUNNEL_CHECK_EVERY = 10           # 公開URL疎通は 10 分ごと（外部リクエスト削減）

# ---- 死活監視 / 後片付けの能力表（共有セルが読む）----
ENGINE_LOG = "/tmp/ollama.log"
# ollama は -f だと子の `llama-server -m ...ollama_models...` まで全一致して
# 偽 alive になる。プロセス名完全一致で見る。proxy は argv で見る。
ENGINE_PROCS = [("ollama", "exact"), ("auth_proxy.py", "argv")]
CLEANUP_TARGETS = [
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-x ollama", "ollama"),
    ("-f llama-server", "llama-server"),
]


def engine_cleanup():
    """VRAM を返してから落とす（OLLAMA_KEEP_ALIVE=-1 で固定されるため）。

    これをやらないと 23 GiB が VRAM に残ったままカーネルが COMPLETE になり、
    GPU を掴み続ける（実測: expires_at が 2319 年になる）。プロセスを落としても
    Kaggle 側の枠開放は保証されないが、少なくとも VRAM は即座に返る。
    """
    unloaded = kc_unload_model(PORT, MODEL_NAME)
    kc_wait_vram_free(PORT)
    return unloaded


# ---- Ollama の起動環境変数 ----
# 起動セルと確認セルが共有するので、ここで 1 回だけ定義する。
# OLLAMA_CONTEXT_LENGTH : コンテキスト長（未設定だと 4096 に落ちる）
# OLLAMA_KV_CACHE_TYPE  : KV を q8_0 に量子化して VRAM を半分に
# OLLAMA_FLASH_ATTENTION: 有効化
# OLLAMA_KEEP_ALIVE     : -1 で常時ロード維持
OLLAMA_ENV = {
    **os.environ,
    "OLLAMA_MODELS": OLLAMA_MODELS,
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_ORIGINS": "*",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_FLASH_ATTENTION": "1",
    # 既定 5分（stall 検出）。v9 は load 198s で通過したが v10 は超過して
    # "timed out waiting for llama-server to start" → 500 になった。
    # mmap 読込はページキャッシュ次第で揺れるので 10分に緩和（warmup 900s 未満）。
    "OLLAMA_LOAD_TIMEOUT": "600",
    "OLLAMA_NUM_PARALLEL": "1",   # qwen35 は並列が blocklist されているため 1
}

# ---- 検証成果物の永続化 ----
# `kaggle kernels output` が zip 化するのは **/kaggle/working** のだけ。
# /output は公式の永続先ではなく、作れても zip に載らないことがある
# （= 保存したつもりで証跡ゼロ）。だから両方へ書く。
OUT_DIRS = KC_OUT_DIRS
save_output = kc_save_output
fail = kc_fail

print()
print("=" * 56)
print(f"  認証        : {'OFF（URL を知れば誰でも利用可）' if not PROXY_AUTH['API_PASS'] else 'ON'}")
print(f"  公開ポート  : {PROXY_PORT}（Ollama 本体は {PORT}）")
print("=" * 56)
