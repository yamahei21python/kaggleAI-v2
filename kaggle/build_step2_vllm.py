#!/usr/bin/env python3
"""Step2 notebook builder (vLLM / AWQ+MTP): Qwen3.8-27B-Uncensored on Kaggle T4 x2

既存の Ollama 版（build_step2.py）とは**別ファイル・別 kernel**。
実証済みのステージング処理（プロキシ / UI / トンネル / Gist / keepalive）は
そのまま流用し、推論エンジン部分だけ vLLM に差し替える。

    python3 kaggle/build_step2_vllm.py                  # cloudflared
    STEP2_TUNNEL=ngrok python3 kaggle/build_step2_vllm.py

出力: kaggle/step2-vllm-kernel/step2-vllm-qwen38-27b-<tunnel>.ipynb
      （push はしない。Dataset 作成完了後に手動で push する）
"""
import os
import pathlib
import sys

# ── 固定パラメータ ──────────────────────────────────────────
OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
DATASET_ID = f"{OWNER}/{DATASET_SLUG}"
HF_MODEL = "twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP"
SERVED_NAME = "qwen3.8-27b-awq"

TUNNEL = os.environ.get("STEP2_TUNNEL", "cloudflared").strip().lower()
if TUNNEL not in ("cloudflared", "ngrok"):
    raise SystemExit(f"STEP2_TUNNEL={TUNNEL!r} は未対応（cloudflared / ngrok）")
IS_NGROK = TUNNEL == "ngrok"

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kbuild import embed, metadata, pack_nb  # noqa: E402
from runtime import source_bundle  # noqa: E402
# prepare-vllm-dataset.ipynb / kernel-metadata.json と同じディレクトリに置くと
# メタデータが衝突するので、kernel 用は専用ディレクトリへ分ける。
# トンネルごとにディレクトリを分ける。kernel-metadata.json は push 時の
# 必須ファイル名なので、同居させると片方の設定が上書きされる。
OUT_DIR = HERE / "step2-vllm-kernel" / TUNNEL
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"step2-vllm-qwen38-27b-{TUNNEL}.ipynb"

# 単一ソース（kaggle/step2/*）をそのまま埋め込む。notebook 側で二重管理しない。
PROXY_SRC_B64 = embed.b64_file(HERE / "step2" / "auth_proxy.py")
CHAT_SRC_B64 = embed.b64_file(HERE.parent / "llama-chat.html")

cells = []


def _cell(cell_type, source):
    # nbformat 5 では id 必須（無いと MissingIDFieldWarning が出る）
    d = {"cell_type": cell_type, "id": f"cell{len(cells):02d}", "metadata": {},
         "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


def _add(cell_type, source):
    cells.append(_cell(cell_type, source))


def _tpl(src, **kw):
    """__KEY__ 置換。f-string だと notebook 側の dict リテラルを
    すべて二重 brace にする必要があり事故るので、置換で渡す。"""
    for k, v in kw.items():
        src = src.replace(f"__{k}__", str(v))
    return src


# ── 0: title ────────────────────────────────────────────────
cf_bullet = ("" if IS_NGROK else
             "   - `my-cloudflared-bin`（cloudflared 40 MB）"
             "— 無いと毎回 GitHub から落とす")
secret_note = (
    """> ✅ **必要な Secrets**: `NGROK_TOKEN`（必須）
> `NGROK_DOMAIN` はノートに埋め込み済み → **毎回同じ URL**。
>
> 🔓 認証はありません（URL を知れば誰でも使えます）。
> 他人に使われたくないときは環境変数 `API_PASS` を設定してください。"""
    if IS_NGROK else
    """> ✅ **Secrets の設定は不要です。**（cloudflared はアカウント不要）
>
> 🔓 認証はありません（URL を知れば誰でも使えます）。
> 他人に使われたくないときは環境変数 `API_PASS` を設定してください。
>
> ⚠ 公開URLは **実行のたびに変わります**（quick tunnel の仕様）。""")

_add("markdown", _tpl("""# 🚀 Step 2 (vLLM): Qwen3.8-27B AWQ + __TUNNEL__ 公開APIサーバー

`__HFMODEL__` を **vLLM** で T4 x2 に載せ、__TUNNEL__ で公開します。

---

### 前提条件（これだけ）

1. **Accelerator**: `GPU T4 x2`
2. **Internet**: `On`
3. **Add Data**: 以下を追加
   - `__DATASET__`（AWQ 4bit モデル 約19 GB）
__CFBULLET__
__SECRET__

---

### Ollama 版との違い

| 項目 | Ollama 版 | このノート（vLLM） |
|---|---|---|
| 量子化 | GGUF Q4_K_M | AWQ W4A16 |
| 推論 | llama.cpp | vLLM（PagedAttention） |
| MTP | なし | `--speculative-config` で有効化（失敗時は自動で無効） |
| 速度 | 13.7 t/s 実測 | T4 x2 では 10〜25 t/s 見込み |

> ℹ T4 は sm75 のため **Marlin カーネルと FP8 KV cache は使えません**。
> AWQ は Triton 実装にフォールバックします。
""", TUNNEL=TUNNEL, HFMODEL=HF_MODEL, DATASET=DATASET_ID,
            CFBULLET=cf_bullet, SECRET=secret_note))

# ── 0.5: 共通ランタイム ─────────────────────────────────────
# proxy 起動・トンネル URL 取得・keepalive conf・後片付けを runtime/ に
# 一本化する。手で編集しない（真実源はリポジトリ直下の runtime/*.py）。
_add("markdown", "---\n### 🧩 共通ランタイム（proxy / tunnel / keepalive / verify）\n\n"
                 "`runtime/` の真実源をビルダーが連結して埋め込んでいます。\n")
_add("code", source_bundle("net", "proc", "storage", "keepalive", "proxy"))

# ── 1: 設定 ─────────────────────────────────────────────────
_add("markdown", "---\n### ⚙️ 設定（Dataset 検出 → VRAM 計測 → 起動条件決定）\n")

_add("code", _tpl('''# ============================================================
# Step 2 (vLLM): 共通設定
# ============================================================
import glob as _glob
import os


def _find_input(slug):
    """Dataset のマウント場所を探す。

    Kaggle は /kaggle/input/<slug> と /kaggle/input/datasets/<owner>/<slug> の
    両方が実在しうる。深さ2まで探索し、最初に「空でない」ものを返す。
    """
    cands = []
    for pat in ("/kaggle/input/" + slug,
                "/kaggle/input/*/" + slug,
                "/kaggle/input/*/*/" + slug):
        cands += [p for p in _glob.glob(pat) if os.path.isdir(p)]
    for p in sorted(cands):
        try:
            if os.listdir(p):
                return p
        except OSError:
            continue
    return cands[0] if cands else "/kaggle/input/" + slug


MODEL_SRC = _find_input("__DATASET_SLUG__")
SERVED_NAME = "__SERVED__"
VLLM_PORT = 8000
PROXY_PORT = 11435
TUNNEL_KIND = "__TUNNEL__"

# Basic 認証プロキシ / UI のソース（ビルダーが埋め込む）
PROXY_SRC_B64 = "__PROXY_B64__"
CHAT_SRC_B64 = "__CHAT_B64__"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

# pip に渡す vLLM 指定（切り戻したいときは "vllm==0.10.2" 等に変える）
VLLM_PIP_SPEC = os.environ.get("VLLM_PIP_SPEC", "vllm==0.32.0")


_gpu_names, _vram_each = kc_detect_gpu()
VRAM_TOTAL = sum(_vram_each)
print(f"GPU      : {_gpu_names}")
print(f"VRAM     : {_vram_each} MiB each  →  合計 {VRAM_TOTAL} MiB")

# /dev/shm は TP>1 の必須資源。vLLM のリングバッファは既定で
# (24 MiB + 数 B) x 10 ブロック ≈ 240 MiB を 1 本で確保する。
# 足りなければ即 RuntimeError になるが、境界値だと起動後に詰まるので記録する。
try:
    _shm = os.statvfs("/dev/shm")
    SHM_FREE_MIB = _shm.f_bavail * _shm.f_frsize // (1 << 20)
    SHM_TOTAL_MIB = _shm.f_blocks * _shm.f_frsize // (1 << 20)
    print(f"/dev/shm : {SHM_TOTAL_MIB} MiB（空き {SHM_FREE_MIB} MiB）"
          f"  cpu={os.cpu_count()}")
    if SHM_TOTAL_MIB < 512:
        print("⚠ /dev/shm が小さい。TP>1 の共有メモリ通信で詰まる可能性がある")
except Exception as _e:
    print(f"/dev/shm : 取得失敗 {_e!r}")

if VRAM_TOTAL == 0:
    raise RuntimeError(
        "GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")
if len(_vram_each) < 2:
    print("⚠ GPU が 1 枚です。Tensor Parallel は 1 に落とします"
          "（27B AWQ は 1 枚に収まらない可能性が高い）")
TP_SIZE = 2 if len(_vram_each) >= 2 else 1

# ---- 起動条件 ----
# 27B AWQ は約 19 GB。T4 x2 = 30 GB のうち 0.90 = 27 GB を使う想定で、
# 残り ~8 GB が KV cache になる。実際に収まるかは config 次第なので、
# 起動失敗時に max-model-len を落として再試行する（serve セル）。
# 2026-10-29 机上計算: 重み 18.20 GiB + KV f16 68 KiB/token。
# 64K=22.45 GiB（予算 27.0 で余裕 4.5）、96K=24.58（余裕 2.4）、
# 128K=26.70（余裕 0.3 = graph/activation で確実に溢れる）。
# T4 は fp8 KV 不可なので Ollama の q8_0（34 KiB/token）のようには詰められない。
# → 上限狙いは 64K。実測で 1〜2 GiB ずれるので 32K/16K も残す。
MAXLEN_CANDIDATES = [65536, 32768, 16384, 8192]
MAX_MODEL_LEN = MAXLEN_CANDIDATES[0]
GPU_MEM_UTIL = 0.90
MAX_NUM_SEQS = 2
KV_CACHE_DTYPE = "auto"        # T4 は FP8 非対応。fp8 を指定してはいけない

# MTP（multi-token prediction）は任意。失敗したら自動で無効にして再起動する。
MTP_ENABLED = os.environ.get("VLLM_MTP", "1") != "0"
MTP_METHOD = os.environ.get("VLLM_SPEC_METHOD", "mtp")
MTP_TOKENS = int(os.environ.get("VLLM_SPEC_TOKENS", "2"))

# ---- 公開ポート ----
API_USER = os.environ.get("API_USER", "") or "ollama"
API_PASS = os.environ.get("API_PASS", "")

# ---- 永続化ヘルパー ----
# `kaggle kernels output` が zip 化するのは /kaggle/working のみ。
OUT_DIRS = KC_OUT_DIRS
save_output = kc_save_output
fail = kc_fail


print()
print("=" * 56)
print(f"  モデル      : {MODEL_SRC}")
print(f"  配信名      : {SERVED_NAME}")
print(f"  TP          : {TP_SIZE}")
print(f"  MTP         : {'ON' if MTP_ENABLED else 'OFF'}")
print(f"  公開ポート  : {PROXY_PORT}（vLLM 本体は {VLLM_PORT}）")
print(f"  認証        : {'ON' if API_PASS else 'OFF（URL を知れば誰でも利用可）'}")
print("=" * 56)

_files = sorted(os.listdir(MODEL_SRC)) if os.path.isdir(MODEL_SRC) else []
print(f"  Dataset 直下: {len(_files)} 件")
for _f in _files[:12]:
    print(f"    - {_f}")
if "config.json" not in _files:
    fail("config.json が Dataset 直下にありません（Add Data を確認）")
if "model.safetensors" not in _files:
    fail("model.safetensors が Dataset 直下にありません（Add Data を確認）")
''', DATASET_SLUG=DATASET_SLUG, SERVED=SERVED_NAME, TUNNEL=TUNNEL,
            PROXY_B64=PROXY_SRC_B64, CHAT_B64=CHAT_SRC_B64))

# ── 2: install ──────────────────────────────────────────────
_add("markdown", "---\n### 📡 vLLM インストール + プロキシ配置\n\n"
                 "vLLM の wheel 取得に 5〜10 分かかります（2回目以降は Kaggle の"
                 "キャッシュで短縮）。\n")

_add("code", '''# vLLM をインストールし、プロキシと UI を書き出す
import base64
import os
import shutil
import subprocess
import sys

BIN_DIR = "/usr/local/bin"
MIN_SIZE = 1024 * 1024

# ---- cloudflared（Dataset キャッシュ優先）----
if TUNNEL_KIND != "ngrok":
    CF_PATH = _find_input("my-cloudflared-bin")
    dst = os.path.join(BIN_DIR, "cloudflared")
    src = None
    for cand in (os.path.join(CF_PATH, "cloudflared"),
                 os.path.join(CF_PATH, "bin", "cloudflared")):
        if os.path.isfile(cand) and os.path.getsize(cand) >= MIN_SIZE:
            src = cand
            break
    if src:
        shutil.copy2(src, dst)
        subprocess.run(["chmod", "+x", dst])
        print(f"✅ cloudflared 復元: {src}")
    elif not (os.path.isfile(dst) and os.path.getsize(dst) >= MIN_SIZE):
        print("⬇️  cloudflared をダウンロード中...")
        subprocess.run(
            "curl -fsSL --connect-timeout 15 -o /usr/local/bin/cloudflared "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared",
            shell=True, check=False, timeout=300,
        )
    if not os.path.isfile(dst):
        fail("cloudflared がありません（Dataset 未マウント / 取得失敗）")
    r = subprocess.run([dst, "--version"], capture_output=True, text=True)
    print((r.stdout or r.stderr).strip())
    if r.returncode != 0:
        fail("cloudflared の実行に失敗しました（実行ビット/Dataset を確認）")

# ---- vLLM ----
# wheelhouse Dataset（vllm 0.32.0 + 依存 182 wheel / 2.0 GB）があれば
# ネットワークを使わない。実測 install は 336〜512 秒 → 60〜120 秒になる。
# --dir-mode zip で作った Dataset は /kaggle/input に zip 1 個で置かれる。
# その場合は /kaggle/tmp へ 1 回だけ展開する（数十秒。10 分の install より速い）。
_wh = _find_input("vllm-wheelhouse")
_pip_extra = []

def _whl_dir(root):
    w = _glob.glob(os.path.join(root, "*.whl"))
    if w:
        return root
    for _s in sorted(_glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(_s) and _glob.glob(os.path.join(_s, "*.whl")):
            return _s
    return None

if _wh:
    _d = _whl_dir(_wh)
    if _d is None:
        _zips = (_glob.glob(os.path.join(_wh, "*.zip"))
                 + _glob.glob(os.path.join(_wh, "*", "*.zip")))
        if _zips:
            import time as _t
            import zipfile as _zf
            _dst = "/kaggle/tmp/whl"
            os.makedirs(_dst, exist_ok=True)
            _t0 = _t.time()
            print(f"📦 wheelhouse zip を展開: {[os.path.basename(z) for z in _zips]}")
            for _z in _zips:
                with _zf.ZipFile(_z) as _f:
                    _f.extractall(_dst)
            print(f"   展開 {_t.time() - _t0:.1f}s")
            _d = _whl_dir(_dst) or _dst
    if _d:
        _n = len(_glob.glob(os.path.join(_d, "*.whl")))
        print(f"📦 ローカル wheelhouse を使用: {_d}（{_n} wheel）")
        _pip_extra = ["--no-index", "--find-links", _d]
if not _pip_extra:
    print("ℹ wheelhouse Dataset なし → PyPI から取得（5〜10分）")

# ---- warm cache（任意 Dataset）----
# torch.compile の AOT 成果物と Triton JIT を Dataset から復元する。
# コンテナは毎回まっさらなので、これがあると compile 30〜60s を省ける。
# キャッシュキーが違えば単にミスするだけ（壊れない）。
_wc = _find_input("vllm-warm-cache")
if _wc:
    import tarfile as _tf2
    _tars = _glob.glob(os.path.join(_wc, "*.tar")) + _glob.glob(os.path.join(_wc, "*", "*.tar"))
    for _t in _tars:
        try:
            _t0 = __import__("time").time()
            with _tf2.open(_t) as _f:
                _f.extractall("/")
            print(f"🧠 キャッシュ復元: {os.path.basename(_t)} "
                  f"{os.path.getsize(_t)/1e6:.0f} MB ({__import__('time').time()-_t0:.1f}s)")
        except Exception as _e:
            print(f"⚠ キャッシュ復元をスキップ: {_e}")
else:
    print("ℹ warm cache なし（compile をやり直す）")

_r = subprocess.run([sys.executable, "-c", "import vllm"],
                    capture_output=True, text=True)
if _r.returncode == 0:
    print("✅ vLLM は既にインストール済み")
else:
    print(f"⬇️  vLLM をインストール中: {VLLM_PIP_SPEC}（5〜10分）")
    _p = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-input",
         "--no-deps"] + _pip_extra + [VLLM_PIP_SPEC],
        capture_output=True, text=True, timeout=3600,
    )
    print((_p.stdout or "")[-2000:])
    print((_p.stderr or "")[-2000:])
    if _p.returncode != 0:
        fail("vLLM のインストールに失敗しました（上のログ参照）")

# torchaudio は transformers が import する。vLLM の pip install で torch だけが
# 差し替わると CUDA 版数が不一致になり、vllm CLI が即死する
# （実測: vllm 0.32.0 / torch 2.13.0+cu130 に対し torchaudio 2.8.3+cu128）。
#
# 検出は「実際に落ちる経路」で行う。`import vllm.entrypoints.cli.main` だけでは
# 不十分だった（v2 実測）: benchmark.main 等は main() の中で遅延 import される
# ため、深い依存（transformers → torchaudio）の不整合を見逃す。
# torchaudio の整合は serve より前に単独で確認する（v2 の真因）。
# PyPI の torchaudio 2.11.0 は +cu130 ビルド（wheel の version.py で実測）なので、
# torch 2.13.0+cu130 とそのまま一致する。
def _ta_check():
    r = subprocess.run([sys.executable, "-c", "import torchaudio"],
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0, ((r.stdout or "") + (r.stderr or ""))[-900:]


_ta_good, _ta_err = _ta_check()
if not _ta_good:
    print("⚠ torchaudio 不整合 → 2.11.0(+cu130) を --no-deps で入れ直す")
    print(_ta_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                   + _pip_extra
                   + ["--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
                   capture_output=True, text=True, timeout=1800)
    _ta_good, _ta_err = _ta_check()
if not _ta_good:
    # vLLM が torchaudio を使うのは音声モデルだけ（midashenglm/audio.py 等）。
    # このモデルは音声不要なので、外すのが最終手段として妥当。
    print("⚠ まだ不整合 → torchaudio を削除")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ta_good, _ta_err = _ta_check()
print(f"torchaudio import: {'OK' if _ta_good else 'NG'}")

_PREFLIGHT_CMD = ["vllm", "serve", "--help"]


def _cli_ok():
    r = subprocess.run(_PREFLIGHT_CMD, capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout or "") + (r.stderr or "")
    bad = ("Traceback" in out) or ("RuntimeError" in out) or len(out) < 500
    return (r.returncode == 0 and not bad), out[-1500:]


_ok, _err = _cli_ok()
if not _ok:
    # transformers 側は ImportError を握るので、外すのが最短（通信不要）。
    # vLLM 0.32 は torchaudio==2.11.0 を要求する（METADATA 実測）。
    # 版を揃えて入れ直すのが本筋。駄目なら外す（transformers は ImportError を握る）。
    print("⚠ vLLM CLI が起動しない → torchaudio を 2.11.0 で入れ直す")
    print(_err)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input"]
                   + _pip_extra
                   + ["--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
                   capture_output=True, text=True, timeout=1800)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を外して再試行")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                    "torchaudio"], capture_output=True, text=True, timeout=600)
    _ok, _err = _cli_ok()
if not _ok:
    print("⚠ まだ失敗 → torchaudio を torch に合わせて入れ直す")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "-U", "torchaudio"], capture_output=True, text=True,
                   timeout=3600)
    _ok, _err = _cli_ok()
if not _ok:
    # help が壊れていても serve が壊れているとは限らない（実測: CPU では
    # transformers ではなく arg_utils 側で別の失敗をする）。ここで止めず、
    # 起動を試す。壊れていれば最初の試行が即座に落ちるだけで済む。
    print("⚠ vLLM CLI の確認に失敗 → そのまま起動を試す")
    print(_err)

try:
    import torch
    import vllm
except Exception as _e:
    # import 失敗は「torch の入れ替わり」で起きやすい。原因を残して止める。
    fail(f"vllm/torch の import に失敗: {type(_e).__name__}: {_e}")
print(f"✅ vLLM {vllm.__version__} / torch {torch.__version__} "
      f"/ CUDA {torch.version.cuda}")
print(f"   GPU 認識: {torch.cuda.device_count()} 枚")

# ---- プロキシ + UI ----
with open(PROXY_PATH, "wb") as f:
    f.write(base64.b64decode(PROXY_SRC_B64))
with open(CHAT_PATH, "wb") as f:
    f.write(base64.b64decode(CHAT_SRC_B64))
print(f"✅ auth_proxy.py: {os.path.getsize(PROXY_PATH):,} B")
print(f"✅ llama-chat.html: {os.path.getsize(CHAT_PATH):,} B")
''')

# ── 3: proxy ────────────────────────────────────────────────
_add("markdown", "---\n### 🌐 リバースプロキシ起動（UI 配信 + 同一オリジン化）\n\n"
                 "vLLM 本体は 8000、公開は 11435。`/v1/*` をそのまま中継します。\n")

_add("code", '''# リバースプロキシを起動（UI を / で配信し、API を同一オリジンにする）
import os

proxy_env = {
    **os.environ,
    # 認証は既定で無効（API_PASS を設定したときだけ有効）
    "API_USER": API_USER,
    "API_PASS": API_PASS,
}
# クライアントが temperature=0 / repetition_penalty 未指定で送っても
# 崩壊しないよう、サーバ側で下限を矯正する（RP>=1.05 / temp>0）。
proxy_proc = kc_start_proxy(
    PROXY_PORT, f"http://127.0.0.1:{VLLM_PORT}", CHAT_PATH, PROXY_PATH,
    extra_args=("--enforce-sampling",), env=proxy_env, fail=fail)

# vLLM はまだ起動していないので 502 で正常（401 なら設定ミス）
kc_assert_no_auth(PROXY_PORT, path="/v1/models", fail=fail)
''')

# ── 4: vllm serve ───────────────────────────────────────────
_add("markdown", '''---
### 🔥 vLLM サーバー起動（初回は 5〜10 分）

起動条件は **上から順に試行**し、失敗したら次へ落ちます。

| # | MTP | max-model-len | 備考 |
|---|---|---|---|
| 1 | ON | 32768 | 本命 |
| 2 | ON | 16384 | KV 不足時 |
| 3 | OFF | 16384 | MTP 非対応アーキのとき |
| 4 | OFF | 8192 | 最小構成 |
| 5 | OFF | 8192 | V0 エンジン（T4 用フォールバック） |

`--speculative-config` などのフラグは `vllm serve --help` を見て
**存在するものだけ**付けます（バージョン差で即死しないため）。
''')

_add("code", '''# vLLM 起動 → /v1/models が返るまで待機（失敗時は条件を落として再試行）
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request

STAGE = "/kaggle/tmp/vllm-model"
MTP_FILE = "model-mtp.safetensors"


def _stage_model(with_mtp):
    """モデルを symlink で 1 か所にまとめる（19 GB をコピーしない）。

    MTP の draft head (model-mtp.safetensors) は、speculative を切った状態で
    読ませると unexpected key になるため、その場合だけ除外する。
    """
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)          # symlink のみなので実体は消えない
    os.makedirs(STAGE, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(MODEL_SRC)):
        if fn.startswith("."):
            continue
        if fn == MTP_FILE and not with_mtp:
            continue
        src = os.path.join(MODEL_SRC, fn)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(STAGE, fn))
        n += 1
    return n


DRAFT_DIR = "/kaggle/tmp/vllm-draft"
# draft に必要なのは mtp の重み + config 類だけ。残すと `*.safetensors` の glob が
# 本体 17.41 GiB まで拾い、draft が同じだけ読み直す（実測 §27）。
DRAFT_KEEP = ("config", "tokenizer", "chat_template", "generation_config",
              "special_tokens_map", "vocab", "merges", "added_tokens")


def _stage_draft():
    """draft (MTP) 専用の最小ディレクトリを作る。戻り値は symlink 本数。"""
    if not os.path.isfile(os.path.join(MODEL_SRC, MTP_FILE)):
        return 0
    if os.path.isdir(DRAFT_DIR):
        shutil.rmtree(DRAFT_DIR)
    os.makedirs(DRAFT_DIR, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(MODEL_SRC)):
        if fn.startswith("."):
            continue
        if fn != MTP_FILE and not fn.startswith(DRAFT_KEEP):
            continue
        src = os.path.join(MODEL_SRC, fn)
        if not os.path.isfile(src):
            continue
        os.symlink(src, os.path.join(DRAFT_DIR, fn))
        n += 1
    return n


# ---- 利用可能なフラグを確認（バージョン差の吸収）----
# vLLM 0.32 の `serve --help` は Config Group の一覧しか出さず、
# 個々のフラグは `--help=all` にしか無い（GPU 実測 2026-10-29）。
# 旧実装は 2,134 文字の要約を「完全な help」と誤認し、_has() が全フラグを
# 偽判定 → MTP / LMO / chat-template が全部落ちた。既知フラグの有無で
# 「完全な help か」を判定し、ダメなら VERIFIED_FLAGS に倒す。
def _run_help(arg):
    _p = subprocess.run(["vllm", "serve", arg], capture_output=True,
                        text=True, timeout=180)
    return _p.returncode, (_p.stdout or "") + (_p.stderr or "")


def _looks_complete(text):
    """個々のフラグまで含む help か。要約だけなら False。"""
    return (len(text) > 5000
            and "--tensor-parallel-size" in text
            and "--max-model-len" in text)


_rc, HELP = _run_help("--help=all")
if not _looks_complete(HELP):
    _rc2, _h2 = _run_help("--help")
    if len(_h2) > len(HELP):
        HELP, _rc = _h2, _rc2
# ヘルプ本文を成果物に残す（フラグ判定が外れたときの唯一の手掛かり）
save_output("vllm_help.txt", HELP[:20000])
print(f"  vllm serve --help=all: exit={_rc} {len(HELP)} 文字 "
      f"complete={_looks_complete(HELP)}")
print(HELP[:1200])
HELP_OK = (_rc == 0 and _looks_complete(HELP)
           and "Traceback (most recent call last)" not in HELP)
if not HELP_OK:
    print("⚠ 完全な help が取れない → VERIFIED_FLAGS で判定する")


# vllm 0.32.0 wheel の arg_utils/cli_args を静的に走査して実在を確認したもの。
# （詳細 notes/vllm-cpu-findings.md §22）
VERIFIED_FLAGS = {
    "--trust-remote-code", "--reasoning-parser", "--language-model-only",
    "--limit-mm-per-prompt", "--enable-prefix-caching", "--enforce-eager",
    "--tool-call-parser", "--enable-auto-tool-choice", "--speculative-config",
    "--spec-method", "--kv-cache-dtype", "--enable-log-requests",
    "--chat-template", "--chat-template-content-format", "--default-chat-template-kwargs",
    "--attention-backend", "--max-num-batched-tokens", "--generation-config",
    "--override-generation-config", "--load-format", "--tokenizer",
}


def _has(flag):
    """ヘルプ本文の「行頭オプション」として存在するか。

    単純な部分文字列一致だと説明文中の言及で誤判定するため行頭で見る。
    ヘルプが取れない環境では楽観側（True）に倒さず、wheel を静的に走査して
    実在を確認した VERIFIED_FLAGS だけを許可する。楽観側だと 0.32 で消えた
    フラグを付けて未知引数で即死する（過去に実際に踏んだ）。
    """
    if not HELP_OK:
        # 事前に wheel を静的解析して「実在する」と確認済みのものだけ許可する。
        # ここを楽観（True）にすると、0.32 で消えた --chat-template のような
        # フラグを付けて即死する（実測で該当）。
        return flag in VERIFIED_FLAGS
    return re.search(r"^\\s*" + re.escape(flag) + r"\\b", HELP,
                     re.MULTILINE) is not None


# 必須フラグ（ここが通らなければ何を試しても無駄）
CORE_ARGS = [
    "vllm", "serve", STAGE,
    "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1",
    "--port", str(VLLM_PORT),
    "--tensor-parallel-size", str(TP_SIZE),
    "--dtype", "half",
    # 量子化の指定は試行ごとに変える（下の PLAN の 7 要素目）。
    # このモデルの quantization_config は format="pack-quantized"（= compressed-tensors）
    # であり、awq 固定は誤り。auto（指定なし）を第一候補にする。
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--gpu-memory-utilization", str(GPU_MEM_UTIL),
    "--kv-cache-dtype", KV_CACHE_DTYPE,
]

# 任意フラグ。バージョンによって引数値が違うことがあるので、
# 後半の試行では落として「フラグが原因の失敗」を切り分ける。
OPT_ARGS = []
if _has("--trust-remote-code"):
    OPT_ARGS.append("--trust-remote-code")
# 0.32 では --disable-log-requests が廃止され --enable-log-requests に反転。
# 省略しても害がないので、存在した場合だけ付ける（誤指定で試行を落とさない）。
if _has("--disable-log-requests"):
    OPT_ARGS.append("--disable-log-requests")
if _has("--reasoning-parser"):
    # Qwen3 系の思考タグを reasoning_content に分離する
    OPT_ARGS += ["--reasoning-parser", "qwen3"]
if _has("--enable-auto-tool-choice") and _has("--tool-call-parser"):
    OPT_ARGS += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]

# ---- 試行計画（上から順に）----
# 各要素: (ラベル, speculative config, max-model-len, 追加env, 任意フラグを使うか,
#          追加引数, 量子化引数)
# 量子化は「指定なし（auto）」を第一候補。config.json の quantization_config は
# format="pack-quantized" で、AWQ 形式ではない（cfgdeep 実測）。
SPEC = None
if MTP_ENABLED and _has("--speculative-config"):
    SPEC = {"method": MTP_METHOD, "num_speculative_tokens": MTP_TOKENS}
else:
    print("ℹ --speculative-config 非対応 or MTP=0 → MTP なしで起動")

# draft を最小ディレクトリに向ける（17.41 GiB の二重読みを回避）。
# 失敗しても次の段が draft dir なしで再試行するので、ここは楽観で良い。
SPEC_DIR = None
if SPEC:
    _dn = _stage_draft()
    if _dn:
        SPEC_DIR = {**SPEC, "model": DRAFT_DIR}
        print(f"  draft dir : {DRAFT_DIR}（symlink {_dn} 個）")
    else:
        print("ℹ mtp ファイルが無い → draft は target と同じ dir を使う")

# 注意: 0.32 に V0 エンジンは無く、VLLM_USE_V1 / VLLM_ATTENTION_BACKEND は
# envs.py に定義が無い（= 未知 env として警告されるだけ。設定しても無効）。
# T4(sm75) では FlashAttention backend が capability>=8.0 を要求して弾かれ、
# 既定で TRITON_ATTN が選ばれる（triton_attn.supports_compute_capability=True）。
# よってフォールバック段は「明示的に TRITON_ATTN を指定する」に置き換える。

# 視覚塔を落としてテキスト専用で起動する（全段の既定）。
#  - テキスト配信しかしないので不要。0.86 GiB と mm profiling の VRAM を節約。
#  - 実装は StageMissingLayer で子モジュール登録されないため未使用キー警告も出ない
#    （models/utils.py:756 "Don't register this as a child module in order to
#      avoid missing keys when loading weights"）。
# なお quantization_config.ignore の `model.visual.*` は「prefix 文字列」と
# 完全一致で照合される（should_ignore_layer → is_equal_or_regex_match）。
# 視覚塔の prefix は `model.visual...` なので一致する（keymap の key 比較では
# 0 件だったが、判定に使うのは key ではなく prefix）。`re:.*mtp.*` も同様に
# draft の prefix `mtp.*` に一致する。
LMO = ["--language-model-only"] if _has("--language-model-only") else []
if not LMO:
    print("ℹ --language-model-only 非対応 → 視覚塔ありで起動")

AUTO_Q = []                                    # 指定なし（config から自動判定）
CT_Q = ["--quantization", "compressed-tensors"]  # 明示（pack-quantized）
# --quantization awq / auto は config の compressed-tensors と不一致で
# ModelConfig の検証に落ちる（cfgresolve 実測）。試行段からは外した。

STARTUP_BUDGET = int(os.environ.get("VLLM_STARTUP_BUDGET", "1800"))   # 30 分
ATTEMPT_TIMEOUT = int(os.environ.get("VLLM_ATTEMPT_TIMEOUT", "900"))  # 1 試行 15 分
# 10/31 のリセット後（30h）前提の値。install は wheelhouse Dataset で
# 60〜120s になる見込み（未実測）。短い予算で回すときは環境変数で下げる。
# 注意: 残クォータが 30 分未満のときにこの設定で push しないこと。

PLAN = []
# 予算 1500s で 1 試行 600s。MTP を 2 段続けると失敗時に予算を食い潰すので、
# 「MTP+draft dir」→ 失敗したら MTP を一旦あきらめて素の起動を優先する。
if SPEC_DIR:
    PLAN.append(("mtp+dd", SPEC_DIR, MAXLEN_CANDIDATES[0], {}, True, LMO, AUTO_Q))
# ここからは任意フラグを落として最小構成で試す
PLAN.append(("plain", None, MAXLEN_CANDIDATES[0], {}, False, LMO, AUTO_Q))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[1], {}, False, LMO, AUTO_Q))
PLAN.append(("plain", None, 16384, {}, False, LMO, AUTO_Q))
# 2026-10-29 GPU 実測: 32k は torch.compile(61s) 完了後に
# "No available shared memory broadcast block found" を 60s 間隔で出し続け、
# 420s で打ち切った（= ハング or CUDA graph capture/ダミー実行が異常に遅い）。
# eager は graph capture を丸ごと飛ばすので、仮説の切り分けを最優先で入れる。
PLAN.insert(1, ("eager", None, MAXLEN_CANDIDATES[0], {},
                False, LMO + ["--enforce-eager"], AUTO_Q))
# 量子化の明示指定（auto が外した場合）
PLAN.append(("ct", None, 16384, {}, False, LMO, CT_Q))
# attention backend を明示（auto でも TRITON_ATTN になるが、選択を固定する）
PLAN.append(("triton", None, 8192, {}, False,
             LMO + ["--attention-backend", "TRITON_ATTN"], AUTO_Q))
# （eager は上の insert で 2 番目に移した。重複させない）
# MTP の再挑戦（draft dir 無し = 17 GiB を読み直すので最後に置く）
if SPEC:
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[2], {}, True, LMO, AUTO_Q))
# language-model-only が原因で落ちた場合の退避。視覚塔は積むが modality を 0 に
# して実体化させない（get_limit_per_prompt=0 → 同じく StageMissingLayer 経路）。
PLAN.append(("vl0", None, 8192, {}, False,
             ["--limit-mm-per-prompt", '{"image":0,"video":0}'], AUTO_Q))
assert all(len(p) == 7 for p in PLAN), "PLAN の要素数が 7 でない"

# 予算が短いとき（< 900s）は「最も軽い仮説検証」を先頭に持ってくる。
# 実測: 27B の重みロードだけで約 250s、torch.compile が 61s。
# 64K+MTP+グラフを試す余裕は無いので、eager（graph capture 無し）で
# 「そもそも起動できるのか」を先に確定させる。
if STARTUP_BUDGET < 900:
    PLAN.sort(key=lambda _p: 0 if _p[0] == "eager" else 1)
    print("ℹ 起動予算が短い → eager を先頭に並べ替え")


def _gpu_used_mib():
    """全 GPU の使用 VRAM 合計（MiB）。取れなければ -1。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
        return sum(int(x) for x in re.findall(r"[0-9]+", out))
    except Exception:
        return -1


def _kill(proc):
    """vLLM のプロセス木ごと落とし、VRAM が実際に空くまで待つ。

    2026-10-29 実測: 試行 1 を terminate した直後に試行 2 を起動したところ、
    125 秒で "Engine core initialization failed. See root cause above."
    で即死した（EngineCore / Worker が残って VRAM を掴んでいた）。
    SIGTERM は API サーバーにしか届かず、孫（Worker_TP*）は残る。
    """
    try:
        _pgid = os.getpgid(proc.pid)
        os.killpg(_pgid, signal.SIGTERM)
        proc.wait(timeout=20)
        os.killpg(_pgid, signal.SIGKILL)   # 残った子を確実に落とす
    except Exception:
        pass
    for pat in ("VLLM::EngineCore", "vllm serve", "VLLM::WorkerTP"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    # /dev/shm の残骸を掃除する。SharedMemory が unlink されずに残ると
    # 次の試行の check_shm_free_space で Engine core が即死する
    # （実測: attempt 2/3 が 125 秒で "Failed core proc(s): {}"、
    #   同時に resource_tracker が /psm_* のリークを警告）。
    _left = 0
    try:
        for _fn in os.listdir("/dev/shm"):
            if _fn.startswith(("vllm", "psm_")):
                try:
                    os.unlink(os.path.join("/dev/shm", _fn))
                    _left += 1
                except Exception:
                    pass
    except Exception:
        pass
    if _left:
        print(f"  🧽 /dev/shm の残骸 {_left} 件を削除")
    # 解放待ち（最大 90 秒）。ここを飛ばすと次の試行の失敗理由が
    # 「前のプロセスの残骸」になり、切り分けが壊れる。
    for _ in range(18):
        _u = _gpu_used_mib()
        if 0 <= _u < 2000:
            break
        time.sleep(5)
    print(f"  🧹 停止後の VRAM 使用量: {_gpu_used_mib()} MiB")


def _models(timeout=8):
    """起動済みなら /v1/models の JSON を返す（未起動なら None）。"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=timeout
        ) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _smoke(attempts=2, timeout=90):
    """起動後に 1 回だけ生成させ、初回 forward が通るか確かめる。

    /v1/models が返るだけでは不十分。量子化カーネル（Marlin/Exllama）や
    GDN の Triton カーネルは **推論して初めて** 落ちることがあり、その場合
    サーバーは 500 を返し続ける。ここで落とせば次の段に進める。
    """
    body = json.dumps({
        "model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 16,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    last = ""
    for i in range(attempts):
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            txt = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
            return True, (f"tok={(d.get('usage') or {}).get('completion_tokens')} "
                          f"fin={ch.get('finish_reason')} "
                          f"text={txt[:40]!r}")
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            last = f"HTTP {exc.code}: {detail}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        print(f"    smoke {i+1}/{attempts} NG: {last[:200]}")
        time.sleep(3)
    return False, last


proc = None
argv_used = None
started = False
# 起動中の進捗を Gist に流す。Kaggle の UI ログは長大で貼りにくく、
# `kaggle kernels output` は完了まで何も返さないため、外から見える場所が要る。
_GIST_ID = (os.environ.get("GIST_ID") or
            "2b44ac8cb115012a31d3c46661af933f")
_GH_TOKEN = [None]


def _gist_status(text):
    """起動状況を Gist（vllm_status.txt）に書く。失敗しても握りつぶす。"""
    if _GH_TOKEN[0] is None:
        _GH_TOKEN[0] = os.environ.get("GITHUB_TOKEN", "")
        if not _GH_TOKEN[0]:
            try:
                from kaggle_secrets import UserSecretsClient
                _GH_TOKEN[0] = (UserSecretsClient()
                                .get_secret("GITHUB_TOKEN") or "")
            except Exception:
                _GH_TOKEN[0] = ""
    if not _GH_TOKEN[0]:
        return
    body = json.dumps(
        {"files": {"vllm_status.txt": {"content": text}}}).encode()
    req = urllib.request.Request(
        "https://api.github.com/gists/" + _GIST_ID,
        data=body, method="PATCH",
        headers={"Authorization": "Bearer " + _GH_TOKEN[0],
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception:
        pass
# GPU クォータは有限。ラダーを全部試すと 1 時間超になり、途中で打ち切られると
# 成果物が残らない。総予算を切って、必ずログを残して終わるようにする。
# クォータ残 39 分（2,395s）で pip install に 5 分前後かかる前提の値。
# 25 分使うと公開できたとしても数分で打ち切られるため短くしてある。
# 2026-10-29 実測: 27B ハイブリッド（GDN/Mamba）を T4 x2 で起動すると、
# torch.compile 61s の後に "No available shared memory broadcast block" が
# 60s 間隔で出続け、420s では足りなかった（ハングか graph capture が異常に遅い）。
# **クォータ 30h のリセット後は 1 試行 15 分で回す**。
# 10/28 の残 39 分では 720/420 に落として使った（その値では足りなかった）。
_attempts = []
_prev_sig = None
_t_start = time.time()
tail = ""          # 予算切れで 1 度も走らなかった場合に備えて初期化
last_log = ""
smoke_fail = ""    # 起動はしたが生成で落ちた場合の HTTP 応答（ログに出ない）
save_output("vllm_attempts.txt", "起動試行ログ（追記保存）\\n")

for idx, (label, spec, maxlen, env_extra, use_opt, extra_args,
          quant_args) in enumerate(PLAN, 1):
    _elapsed = time.time() - _t_start
    if _elapsed > STARTUP_BUDGET:
        print(f"⏱ 起動予算 {STARTUP_BUDGET}s 超過（{int(_elapsed)}s）"
              f" → 残り {len(PLAN) - idx + 1} 試行を打ち切り")
        break
    n_staged = _stage_model(with_mtp=spec is not None)
    cmd = CORE_ARGS + (OPT_ARGS if use_opt else [])
    cmd += ["--max-model-len", str(maxlen)]
    if spec:
        cmd += ["--speculative-config", json.dumps(spec)]
    cmd += extra_args
    cmd += quant_args
    # tokenizer_config.json に chat_template が無いので明示指定する。
    # 0.32 でも --chat-template は FrontendArgs のフィールドとして実在
    # （entrypoints/launchers/cli_args.py:86。dataclass 経由で自動登録されるため
    #  add_argument の grep では見つからない）。存在しなければ付けない。
    _tpl = os.path.join(STAGE, "chat_template.jinja")
    if os.path.isfile(_tpl) and _has("--chat-template"):
        cmd += ["--chat-template", _tpl]
    log_path = f"/tmp/vllm_{idx}.log"
    last_log = log_path
    # torch.compile / Triton JIT は CPU バウンド。vLLM は既定で 1 スレッドに
    # 落とす（"Reducing Torch threads from 2 to 1 for serving"）ため明示する。
    env_run = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
               **env_extra}
    print()
    print(f"--- 試行 {idx}/{len(PLAN)}: spec={label} "
          f"max_len={maxlen} env={env_extra or 'default'} "
          f"quant={quant_args[1] if quant_args else 'auto'} "
          f"opt_flags={'on' if use_opt else 'off'} "
          f"staged={n_staged} files ---")
    with open(log_path, "w") as log:
        _pre = _gpu_used_mib()
        if _pre > 3000:
            print(f"  ⚠ 起動前の VRAM 使用量 {_pre} MiB（残骸の疑い）→ 掃除")
            for _pat in ("VLLM::EngineCore", "vllm serve"):
                subprocess.run(["pkill", "-9", "-f", _pat],
                               capture_output=True)
            time.sleep(10)
        # start_new_session で独立したプロセスグループにする（_kill で木ごと落とす）
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env_run,
                                start_new_session=True)
    print(f"🔄 vLLM PID: {proc.pid}  log: {log_path}")

    t0 = time.time()
    # 1 試行の上限は残予算でクランプする（最後の 1 段が予算超過で
    # 中途半端に切られ、ログが残らないのを避ける）。
    _tmo = min(ATTEMPT_TIMEOUT,
               max(45, int(STARTUP_BUDGET - (t0 - _t_start))))
    _t_print = 0.0
    while time.time() - t0 < _tmo:
        d = _models(timeout=3)
        if d and d.get("data"):
            started = True
            break
        if proc.poll() is not None:
            break
        _now = time.time() - t0
        if _now - _t_print >= 60:
            _t_print = _now
            # 「起動待ち Ns (0 models)」だけでは何で詰まっているか分からない。
            # ログ末尾をそのまま出す（GPU 実行を 1 回無駄にしないため）。
            try:
                _lns = [ln.strip() for ln in
                        open(log_path, errors="replace").read().splitlines()
                        if ln.strip()]
            except Exception:
                _lns = []
            _last_line = _lns[-1][:170] if _lns else "(ログ空)"
            _gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip().replace("\\n", " ")
            print(f"  ⏳ 起動待ち {int(_now)}s"
                  f"（{len(d.get('data', [])) if d else 0} models）: "
                  f"{_last_line}")
            print(f"     GPU util/mem: {_gpu}")
            _gist_status(
                f"attempt {idx}/{len(PLAN)} label={label} maxlen={maxlen}\\n"
                f"elapsed={int(_now)}s / limit={int(_tmo)}s\\n"
                f"last_log: {_last_line}\\n")
        time.sleep(5)

    smoke_fail = ""
    if started:
        # 起動できても初回 forward でカーネルが落ちる領域（sm75 + 量子化 +
        # GDN Triton）があるため、1 回生成させて通ったときだけ成功とする。
        # smoke の待ち時間も総予算から差し引く（予算超過で次段が走らない事故を防ぐ）。
        _remain = int(STARTUP_BUDGET - (time.time() - _t_start))
        ok, detail = _smoke(timeout=max(30, min(90, _remain)))
        if ok:
            argv_used = cmd
            print(f"✅ vLLM 起動+生成 OK（{int(time.time()-t0)}s, 試行 {idx}）"
                  f" {detail}")
            break
        print(f"⚠ 起動はしたが生成 NG → 次段へ: {detail[:300]}")
        smoke_fail = detail[:800]
        started = False

    tail = ""
    try:
        _full = open(log_path, errors="replace").read()
    except Exception:
        _full = ""
    tail = _full[-4000:]
    # 末尾だけでは「Engine core initialization failed. See root cause above.」
    # のように**原因が上にある**ケースを読めない（2026-10-29 実測）。
    # 失敗ログは丸ごと成果物に残し、原因らしき行も抜き出して表示する。
    if _full:
        save_output(f"vllm_{idx}.log", _full[-3000000:])
    _keys = ("Traceback", "Error", "error", "OutOfMemory", "OOM",
             "RuntimeError", "assert", "CUDA", "not supported",
             "No available shared memory")
    _sig_lines = [ln.strip() for ln in _full.splitlines()
                  if any(k in ln for k in _keys)]
    if _sig_lines:
        print("❌ 起動失敗。原因候補（ログ内の該当行, 先頭20）:")
        for _l in _sig_lines[:20]:
            print("   | " + _l[:200])
    print("❌ 起動失敗。ログ末尾:")
    print(tail)
    # 途中で GPU クォータが尽きても原因が残るよう、失敗ごとに追記保存する。
    _attempts.append(f"--- 試行 {idx} {label} max_len={maxlen} "
                     f"opt={'on' if use_opt else 'off'} ---\\n"
                     + (f"[smoke NG] {smoke_fail}\\n" if smoke_fail else "")
                     + tail)
    save_output("vllm_attempts.txt", "\\n\\n".join(_attempts) + "\\n")
    _kill(proc)
    # 同一エラーで落ち続けるなら、フラグを変えても無駄。予算を残して止める。
    # 末尾 3 行を署名にする。最終行だけだと TypeError 等の一般論で一致して
    # 別原因の失敗まで打ち切ってしまう（残りクォータを捨てる損失が大きい）。
    _last = [ln.strip() for ln in tail.splitlines() if ln.strip()][-3:]
    _sig = " | ".join(_last)[:400]
    # smoke 失敗はサーバーログ末尾が正常行になるため、署名一致で打ち切ると
    # 別段の失敗まで巻き込む。smoke 失敗時は次段へ必ず進む。
    if _sig and _sig == _prev_sig and not smoke_fail:
        print(f"⛔ 同一エラーで再失敗 → ラダー打ち切り: {_sig}")
        break
    _prev_sig = _sig

if not started:
    save_output("vllm_startup.log", tail)
    save_output("vllm_attempts.txt", "\\n\\n".join(_attempts) + "\\n")
    fail("vLLM の起動に失敗しました（vllm_startup.log / vllm_*.log 参照）",
         last_log or "/tmp/vllm_1.log")

# ---- 起動結果を保存 ----
info = _models() or {}
ids = [m.get("id") for m in info.get("data", [])]
max_len = (info.get("data") or [{}])[0].get("max_model_len")
with open("/tmp/vllm_cmd.json", "w") as f:
    json.dump({"argv": argv_used, "spec": bool(SPEC), "models": ids,
               "max_model_len": max_len}, f, indent=2)
print()
print(f"  models        : {ids}")
print(f"  max_model_len : {max_len}")
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.strip())
save_output("vllm_cmd.json", json.dumps(
    {"argv": argv_used, "models": ids, "max_model_len": max_len},
    indent=2) + "\\n")
''')

# ── 5: warmup ───────────────────────────────────────────────
_add("markdown", "---\n### 🔥 ウォームアップ（初回生成は CUDA graph 構築で遅い）\n")

_add("code", '''# ダミー生成で疎通確認 + 速度計測
import json
import subprocess
import time
import urllib.request

payload = {
    "model": SERVED_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
print("⏳ ウォームアップ中...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
except Exception as e:
    # 起動ログは試行番号で名前が変わる。読めなくても失敗自体は伝える。
    for _i in range(1, 7):
        _p = "/tmp/vllm_%d.log" % _i
        try:
            print(f"--- {_p} ---")
            print(open(_p, errors="replace").read()[-2500:])
        except OSError:
            continue
    fail(f"ウォームアップ失敗: {type(e).__name__}: {e}")

msg = (resp.get("choices") or [{}])[0].get("message") or {}
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
usage = resp.get("usage") or {}
ct = usage.get("completion_tokens") or 0
print(f"✅ 応答あり（{dt:.1f}s）")
print(f"   reply : {content[:120]!r}")
print(f"   think : {reasoning[:80]!r}")
print(f"   tokens: completion={ct} total={usage.get('total_tokens')}")
if ct:
    print(f"   速度  : {ct/dt:.1f} tokens/s（ウォームアップ込み）")
if not (content or reasoning or ct):
    fail("生成結果が空です（プロンプト/チャットテンプレートを確認）")

# GPU 常駐（CPU フォールバックでも HTTP 200 は返るので必ず見る）
mem = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.strip()
print(f"   VRAM  : {mem}")
used = [int(x.split(",")[0]) for x in mem.splitlines() if "," in x]
if used and sum(used) < 4000:
    fail(f"VRAM 使用量が少なすぎます（{used} MiB）。CPU 実行の可能性。")
''')

# ── 5.5: 公開URLの取得ヘルパー ───────────────────────────────
# Kaggle の VM は AAAA（IPv6）を返すのに IPv6 経路が無いことがあり、
# `urlopen` が "Network is unreachable" で落ちる。実測（dryrun v6）では
# 同じ公開URLに対して隣接するテストは通り、[1][2] だけ落ちた = 一過性。
# 自分の URL に届かないだけでトンネルは生きているので、IPv4 強制で再試行する。
# keepalive の死活監視がこれで誤判定すると、生きているトンネルを再起動して
# **URL が変わってしまう**（利用者の固定 URL が壊れる）ため必須の対策。
_add("code", '''import socket
import urllib.error
import urllib.request


def _open_public(url, timeout=15):
    """公開URLを開く。失敗したら IPv4 優先で名前解決して 1 回だけ再試行。"""
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except Exception as first:
        err = first
    orig = socket.getaddrinfo

    def _v4(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _v4
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except Exception:
        raise err
    finally:
        socket.getaddrinfo = orig
''')

# ── 6: トンネル ─────────────────────────────────────────────
# 既存 Ollama 版と同一実装（共通インターフェース: TUNNEL_NAME / TUNNEL_LOG /
# RESTART_CMD / probe_url）。keepalive セルがこれらを再利用する。
if IS_NGROK:
    _add("markdown",
         "---\n### 🔗 ngrok トンネル → 公開URL\n\n"
         "Secrets の `NGROK_TOKEN` を使います。`NGROK_DOMAIN` を設定すると\n"
         "**毎回同じ URL** になります（未設定ならランダム）。\n")
    _add("code", '''# ngrok 起動 → 公開URL取得（authtoken は Secrets の NGROK_TOKEN）
import os, subprocess, time

NGROK_BIN = "/usr/local/bin/ngrok"

# patch Dataset には入れていないので取得する（約 30 MB / 数秒）
if not os.path.isfile(NGROK_BIN):
    subprocess.run(
        ["bash", "-c",
         "curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
         " | tar xz -C /usr/local/bin"],
        check=True)

# ---- authtoken ----
_tok = os.environ.get("NGROK_TOKEN", "")
if not _tok:
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("NGROK_TOKEN") or ""
    except Exception:
        _tok = ""
if not _tok:
    fail("NGROK_TOKEN がありません（Kaggle Secrets に登録してください）")
subprocess.run([NGROK_BIN, "config", "add-authtoken", _tok],
               check=True, capture_output=True)

# ---- 固定ドメイン ----
# 優先順: 環境変数 NGROK_DOMAIN → Secrets NGROK_DOMAIN → 埋め込み既定値。
NGROK_DOMAIN_DEFAULT = "rebalance-cornflake-sagging.ngrok-free.dev"
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()
if not NGROK_DOMAIN:
    try:
        from kaggle_secrets import UserSecretsClient
        NGROK_DOMAIN = (UserSecretsClient().get_secret("NGROK_DOMAIN") or "").strip()
    except Exception:
        NGROK_DOMAIN = ""
if not NGROK_DOMAIN:
    NGROK_DOMAIN = NGROK_DOMAIN_DEFAULT
print("📌 固定ドメイン: " + (NGROK_DOMAIN or "（未設定 → ランダムURL）"))

TUNNEL_NAME = "ngrok"
TUNNEL_LOG = "/tmp/ngrok.log"
# --pooling-enabled が無いと、同じ固定ドメインが前回セッションから残っている
# ときに ERR_NGROK_334 で即死する（実測）。
RESTART_CMD = [NGROK_BIN, "http", str(PROXY_PORT),
               "--log=stdout", "--pooling-enabled"]
if NGROK_DOMAIN:
    RESTART_CMD.append("--url=https://" + NGROK_DOMAIN)


def probe_url():
    """ngrok のローカル管理APIから公開URLを取る（ログ解析より確実）。"""
    return kc_probe_url("ngrok")


subprocess.run(["pkill", "-x", "ngrok"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as _log:
    ng_proc = subprocess.Popen(RESTART_CMD, stdout=_log, stderr=_log)
print("🔄 ngrok PID: " + str(ng_proc.pid))

public_url = kc_wait_url("ngrok", TUNNEL_LOG)
if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("ngrok の公開URL取得に失敗しました", TUNNEL_LOG)

print("🌍 URL: " + public_url)
print("⏳ トンネル確立待ち（最大 60s）...")
_elapsed = kc_wait_http(public_url + "/healthz", attempts=20, interval=3,
                        opener=_open_public)
if _elapsed is None:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")
else:
    print("✅ トンネル確立（" + str(_elapsed) + "s）")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\\n")
print()
print("=" * 60)
print("  🌍 PUBLIC URL : " + public_url)
print("=" * 60)
''')
else:
    _add("markdown", "---\n### 🔗 cloudflared トンネル → 公開URL\n")
    _add("code", '''# cloudflared quick tunnel 起動 → 公開URL取得
# アカウント不要。authtoken も Secrets も要らない。
import os, subprocess, time

CF_BIN = "/usr/local/bin/cloudflared"
TUNNEL_NAME = "cloudflared"
TUNNEL_LOG = "/tmp/cloudflared.log"
RESTART_CMD = [CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]


def probe_url():
    """cloudflared のログから公開URLを取る。"""
    return kc_probe_url("cloudflared", TUNNEL_LOG)


# 前回のトンネルを確実に落とす（多重起動すると URL が混ざる）
subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as log:
    cf_proc = subprocess.Popen(RESTART_CMD, stdout=log, stderr=log)
print(f"🔄 cloudflared PID: {cf_proc.pid}")

# ---- 公開URL取得（ログに URL が出るまで待つ）----
public_url = kc_wait_url("cloudflared", TUNNEL_LOG)
if not public_url:
    print("⚠ 公開URLを取得できませんでした")
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("cloudflared の公開URL取得に失敗しました", TUNNEL_LOG)

# ---- トンネル確立まで待つ ----
print(f"🌍 URL: {public_url}")
print("⏳ トンネル確立待ち（最大 60s）...")
_elapsed = kc_wait_http(f"{public_url}/healthz", attempts=20, interval=3,
                        opener=_open_public)
if _elapsed is None:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")
else:
    print(f"✅ トンネル確立（{_elapsed}s）")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\\n")
print()
print("=" * 60)
print(f"  🌍 PUBLIC URL : {public_url}")
print("=" * 60)
''')

# ── 7: Gist（Ollama 版と同一）────────────────────────────────
_add("markdown", "---\n### 📌 固定URL配布（quick tunnel の URL を Gist に反映）\n\n"
                 "quick tunnel の URL は毎回変わるので、**取得先だけ固定**します。\n"
                 "`GITHUB_TOKEN` 未設定ならスキップします。\n")

_add("code", '''# 現在の公開URLを Gist に書き出す（取得先の固定）
# keepalive でトンネルが再起動したときも同じ関数を呼ぶ。
import json, os, urllib.request

GIST_ID_DEFAULT = "2b44ac8cb115012a31d3c46661af933f"
GIST_OWNER = "yamahei21python"
GIST_ID = os.environ.get("GIST_ID", "").strip()
if not GIST_ID:
    try:
        from kaggle_secrets import UserSecretsClient
        GIST_ID = (UserSecretsClient().get_secret("GIST_ID") or "").strip()
    except Exception:
        GIST_ID = ""
if not GIST_ID:
    GIST_ID = GIST_ID_DEFAULT


def _gh_token():
    tok = os.environ.get("GITHUB_TOKEN", "")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN") or ""
    except Exception:
        return ""


def publish_url(u):
    """現在の公開URL を Gist に書く（URL が毎回変わるため）。"""
    tok = _gh_token()
    if not tok:
        print("ℹ GITHUB_TOKEN 未設定 → 固定URL配布はスキップ（接続情報セルを使う）")
        return False
    # Ollama 版は url.txt を使う。同じ gist を共有しても衝突しないよう別名にする。
    files = {"vllm_url.txt": {"content": u + "\\n"}}
    body = json.dumps({"files": files}).encode()
    req = urllib.request.Request(
        "https://api.github.com/gists/" + GIST_ID,
        data=body,
        method="PATCH",
        headers={
            "Authorization": "Bearer " + tok,
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print("✅ 固定URL配布を更新: HTTP " + str(r.status))
            return True
    except Exception as e:
        print("⚠ Gist 更新失敗（続行）: " + repr(e))
        return False


publish_url(public_url)
print("📌 固定取得先: https://gist.githubusercontent.com/"
      + GIST_OWNER + "/" + GIST_ID + "/raw/vllm_url.txt")
''')

# ── 8: verify（vLLM 用に書き換え）────────────────────────────
_add("markdown", "---\n### ✅ 公開URL経由の疎通確認（vLLM 版）\n\n"
                 "UI / models / streaming / OpenAI互換 / GPU常駐 の5点。\n")

_add("code", '''# 公開URL経由の疎通確認（vLLM の OpenAI 互換 API 前提）
import json, time as _time, urllib.request, urllib.error

try:
    with open("/tmp/public_url.txt") as f:
        public_url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

# ---- トンネル確立の取り直し ----
for i in range(10):
    try:
        with _open_public(f"{public_url}/healthz", timeout=10) as r:
            if r.status == 200:
                if i:
                    print(f"✅ トンネル確立をここで確認（{(i+1)*3}s）")
                break
    except Exception:
        pass
    _time.sleep(3)
else:
    print("⚠ /healthz に届きません。トンネル未確立の可能性（[1] で判定）")

lines = [
    f"PUBLIC_URL={public_url}",
    f"MODEL={SERVED_NAME}",
    f"TIME={_time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]
failures = []


def _judge(idx, label, ok, detail):
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _with_retry(fn, tries=6, delay=5):
    """一過性の DNS/経路瞬断を吸収する（HTTP エラーは答えなので再試行しない）。"""
    return kc_retry(fn, tries=tries, delay=delay)


def _post(url, payload, timeout=600, stream=False):
    return kc_post_json(url, payload, timeout=timeout)


CHAT_BODY = {
    "model": SERVED_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
}

# [1] UI
print("[1] 認証なしで UI")
try:
    with _with_retry(lambda: _open_public(
            f"{public_url}/", timeout=60)) as r:
        html = r.read()
        ok_ui = (r.status == 200 and len(html) > 10000
                 and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
        _judge(1, "ui", ok_ui,
               f"HTTP {r.status} {len(html)}B")
except Exception as e:
    _judge(1, "ui", False, f"{type(e).__name__}: {e}")

# [2] /v1/models（vLLM が応答している証明）
print("[2] /v1/models")
try:
    with _with_retry(lambda: _open_public(
            f"{public_url}/v1/models", timeout=60)) as r:
        md = json.loads(r.read())
    ids = [m.get("id") for m in md.get("data", [])]
    _judge(2, "models", r.status == 200 and SERVED_NAME in ids,
           f"HTTP {r.status} ids={ids}")
except Exception as e:
    _judge(2, "models", False, f"{type(e).__name__}: {e}")

# [3] ストリーミング（SSE がトンネルを抜けるか）
print("[3] ストリーミング")
try:
    body = dict(CHAT_BODY, stream=True)
    resp3 = _with_retry(lambda: _post(f"{public_url}/v1/chat/completions",
                                      body, timeout=600))
    times = []
    t0 = _time.time()
    with resp3 as r:
        for raw in r:            # 行到着 = 逐次受信の証拠
            if not raw.strip():
                continue
            times.append(_time.time() - t0)
    total = _time.time() - t0
    if len(times) >= 3:
        first = times[0]
        spread = times[-1] - first
        ok = spread > 0.2 and first < total * 0.8
        detail = (f"chunks={len(times)} first={first:.2f}s "
                  f"total={total:.2f}s spread={spread:.2f}s")
    else:
        ok, detail = False, f"chunks={len(times)} first=N/A"
    _judge(3, "streaming", ok, detail)
except Exception as e:
    _judge(3, "streaming", False, f"{type(e).__name__}: {e}")

# [4] OpenAI 互換（temperature=0 を送っても 200 で生成される）
#     プロキシの --enforce-sampling が 0.7 に矯正する経路の確認でもある。
print("[4] /v1/chat/completions (temperature=0)")
try:
    body4 = dict(CHAT_BODY, temperature=0, repetition_penalty=1.0)
    with _with_retry(lambda: _post(f"{public_url}/v1/chat/completions",
                                   body4, timeout=300)) as r:
        st4 = r.status
        cj = json.loads(r.read())
    ctext = "".join((c.get("message") or {}).get("content") or ""
                    for c in cj.get("choices", []))
    usage = cj.get("usage") or {}
    ok4 = st4 == 200 and bool(ctext or usage.get("completion_tokens"))
    _judge(4, "openai-compat", ok4,
           f"HTTP {st4} reply={ctext[:40]!r} usage={usage}")
except Exception as e:
    _judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

# [5] GPU 常駐（CPU フォールバック検出）
print("[5] GPU 常駐")
try:
    import subprocess as _sp
    mem = _sp.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30).stdout.strip()
    used = [int(x.split(",")[0]) for x in mem.splitlines() if "," in x]
    alive = _sp.run(["pgrep", "-f", "vllm serve"],
                    capture_output=True, text=True).stdout.strip() != ""
    _judge(5, "gpu-resident", alive and bool(used) and sum(used) > 4000,
           f"proc={'alive' if alive else 'DEAD'} vram={used}")
except Exception as e:
    _judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

# [6] 会話（マルチターン: 履歴を渡して文脈が保たれるか）
print("[6] 会話（マルチターン）")


def _reply_parts(j):
    """content と reasoning_content を分けて返す（thinking モデル対策）。"""
    m = (j.get("choices") or [{}])[0].get("message") or {}
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip()


try:
    conv = [{"role": "user",
             "content": "私の名前はコウヘイです。覚えてください。短く返事して。"}]
    with _with_retry(lambda: _post(f"{public_url}/v1/chat/completions",
                                   dict(CHAT_BODY, messages=conv,
                                        max_tokens=96), timeout=300)) as r1:
        j1 = json.loads(r1.read())
    c1, r1 = _reply_parts(j1)
    # thinking だけで content が空のことがある。履歴には本文を優先し、
    # 無ければ思考の先頭を使う（空の assistant を積むと文脈が切れる）。
    conv.append({"role": "assistant", "content": c1 or r1[:600] or "(empty)"})
    conv.append({"role": "user", "content": "私の名前は何？名前だけ答えて。"})
    with _with_retry(lambda: _post(f"{public_url}/v1/chat/completions",
                                   dict(CHAT_BODY, messages=conv,
                                        max_tokens=96), timeout=300)) as r2:
        j2 = json.loads(r2.read())
    c2, r2 = _reply_parts(j2)
    a2 = (c2 + "\\n" + r2).strip()
    name_ok = any(x in a2.lower() for x in ("コウヘイ", "コーヘイ", "kohei"))
    _judge(6, "conversation", bool(c1 or r1) and name_ok,
           f"turn1={(c1 or r1)[:50]!r} turn2={a2[:70]!r}")
except Exception as e:
    _judge(6, "conversation", False, f"{type(e).__name__}: {e}")

# ---- レポート ----
report = "\\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\\n"):
    failures.append("verify_result.txt を永続領域に保存できなかった")

if failures:
    fail("疎通確認に失敗: " + "; ".join(failures), TUNNEL_LOG)
print("✅ 6/6 パス — UI / models / streaming / OpenAI互換 / GPU常駐 / 会話")
''')

# ── 9: 接続情報 ─────────────────────────────────────────────
_add("markdown", "---\n### 📋 接続情報\n")

_add("code", '''# 接続情報（クライアント側の設定にコピペ）
import json

try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

try:
    with open("/tmp/vllm_cmd.json") as f:
        vinfo = json.load(f)
    max_len = vinfo.get("max_model_len")
except Exception:
    max_len = None

opencode_cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "kaggle": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Kaggle vLLM",
            "options": {"baseURL": f"{url}/v1", "apiKey": "dummy"},
            "models": {SERVED_NAME: {"name": "Qwen3.8-27B-Uncensored (AWQ)"}},
        }
    },
    "model": f"kaggle/{SERVED_NAME}",
}

buf = [
    "=" * 64,
    f"  🌍 Public URL : {url}",
    f"  💬 チャット UI : {url}/",
    f"  🤖 Model      : {SERVED_NAME}",
    f"  📏 max_len    : {max_len}",
    "  🔓 認証        : なし（URL を知れば誰でも使える）",
    "  🎛 サンプリング : temperature<=0 → 0.7 / repetition_penalty → 1.05 に矯正",
    "=" * 64,
    "",
    "📋 curl:",
    f"  curl {url}/v1/models",
    "",
    "📋 Python（OpenAI 互換クライアント）:",
    "  from openai import OpenAI",
    "  client = OpenAI(",
    f"      base_url='{url}/v1',",
    "      api_key='dummy',",
    "  )",
    f"  client.chat.completions.create(model='{SERVED_NAME}', messages=[...])",
    "",
    "📋 opencode.json:",
    json.dumps(opencode_cfg, indent=2, ensure_ascii=False),
    "",
    "⚠ quick tunnel: 稼働保証なし。切れたらトンネルセルを再実行",
    "⚠ 公開URLは実行のたびに変わる。再実行後は接続情報を取り直す",
    "⚠ Kaggle: 9時間でカーネルが停止します",
]

report = "\\n".join(buf)
print(report)
if not save_output("connection_info.txt", report + "\\n"):
    fail("connection_info.txt を永続領域に保存できませんでした")
''')

# ── 10: keepalive ───────────────────────────────────────────
_add("markdown", '''---
### 🔋 キープアライブ

既定 **5 分**（検証用）で自動終了します（`KEEPALIVE_MINUTES` で変更、上限 480）。

> 実行が完了しないと `/kaggle/working` の zip が生成されず、ログも成果物も
> 取得できません。検証は必ず完了させてから `kaggle kernels output` で行います。

```bash
kaggle kernels output __SLUG__ -p out/
cat out/verify_result.txt out/connection_info.txt
```
'''.replace("__SLUG__", f"{OWNER}/step2-vllm-qwen38-27b-{TUNNEL}"))

_add("code", '''# キープアライブ: 有限ループ（完了させると /kaggle/working の zip を取得できる）
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

KEEPALIVE_MINUTES = int(os.environ.get("VLLM_KEEPALIVE_MIN", "30"))
# 12 分。最悪 8.5(install) + 12(起動予算) + 1(検証) + 12(keepalive)
# + 1(cleanup) = 34.5 分 < 残 39 分。**終わったら `/stop` で即終了**。
# 本番運用なら 480 に戻す。
# 上限。チャット 1 回で proxy が MINUTES=60 を書くため、上限が無いと
# 「1 回送信 → 60 分に延長」で **残クォータ 39 分を超えて強制終了**される
# （強制終了は成果物の zip が保証されない）。テスト中は 30 分で頭打ちにする。
MAX_MINUTES = 30
TUNNEL_CHECK_EVERY = 10

if KEEPALIVE_MINUTES <= 0:
    fail(f"KEEPALIVE_MINUTES={KEEPALIVE_MINUTES} は無効（1〜{MAX_MINUTES} で指定）")
if KEEPALIVE_MINUTES > MAX_MINUTES:
    print(f"⚠ KEEPALIVE_MINUTES={KEEPALIVE_MINUTES} → {MAX_MINUTES} に制限")
    KEEPALIVE_MINUTES = MAX_MINUTES

KEEPALIVE_CONF = "/tmp/keepalive.conf"
KEEPALIVE_KEEPALIVE = 60      # チャット送信ごとにリセットする時間（分）


def _read_keepalive_conf():
    """RUN=1/0 と MINUTES を読む（既定は 1 と KEEPALIVE_KEEPALIVE）。"""
    run, minutes, _mtime = kc_read_conf(
        KEEPALIVE_CONF, KEEPALIVE_KEEPALIVE, MAX_MINUTES, fail=fail,
        quiet_clamp=True)
    return run, minutes


try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    url = ""


def _proc(name):
    """プロセス生死。vLLM は "vllm serve" の argv で照合する。"""
    return kc_proc_state(name, "argv" if name == "vllm serve" else "exact")


def _gpu():
    return kc_gpu_usage()


print(f"Public URL: {url}")
RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()
if RUN:
    print(f"🔋 ON。{KEEPALIVE_KEEPALIVE}分間維持（チャット送信でリセット、■ で停止）")
else:
    print("🛑 OFF（ RUN=0 ）。即時終了します")
    KEEPALIVE_MINUTES = 0
print("完了後: kaggle kernels output <slug> -p <dir> で成果物を取得")
print()

def _conf_mtime():
    """keepalive.conf の更新時刻。proxy がチャット送信ごとに書き直す。"""
    return kc_conf_mtime(KEEPALIVE_CONF)


log = []
vllm_st = cf_st = "unknown"
cf_restarted = 0
tunnel_dead = 0
_restarted_now = False
# 1周も回らなかった場合（RUN=0 即時終了）は死活判定をしない。
# ループ未実行だと vllm_st/cf_st が "unknown" のままになり、
# 正常に動いているのに fail() する（実測バグ）。
ran_any = False
i = 0
# for ではなく while にする理由: proxy はチャット送信ごとに
# /tmp/keepalive.conf を MINUTES=60 で書き直す。for の range は開始時に
# 固定されるため、延長が効かない（実測バグ）。conf の mtime 変化で延ばす。
limit = KEEPALIVE_MINUTES
conf_mtime = _conf_mtime()
while i < limit:
    i += 1
    RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()
    if not RUN:
        print(f"RUN=0 で {i}周目にキープアライブ終了")
        break
    _m = _conf_mtime()
    if _m != conf_mtime:
        conf_mtime = _m
        limit = min(i - 1 + KEEPALIVE_MINUTES, MAX_MINUTES)
        print(f"♻ アクティビティ検出 → 残り {KEEPALIVE_MINUTES} 分に延長")
    ran_any = True
    time.sleep(60)
    vllm_st, cf_st = _proc("vllm serve"), _proc(TUNNEL_NAME)

    # トンネルが死んだら1回だけ起こす（URL が変わるので取り直す）
    if cf_st == "DEAD" and cf_restarted == 0:
        cf_restarted = 1
        try:
            subprocess.run(["pkill", "-x", TUNNEL_NAME], capture_output=True)
            time.sleep(1)
            with open(TUNNEL_LOG, "w") as _lg:
                subprocess.Popen(RESTART_CMD, stdout=_lg, stderr=_lg)
            _new_url = ""
            for _ in range(30):
                time.sleep(1)
                _new_url = probe_url()
                if _new_url:
                    break
            if _new_url:
                url = _new_url
                with open("/tmp/public_url.txt", "w") as _f:
                    _f.write(url + "\\n")
                save_output("public_url.txt", url + "\\n")
                print(f"♻ {TUNNEL_NAME} 再起動 → 新URL: {url}")
                if "publish_url" in globals():
                    publish_url(url)
                cf_st = "alive(restarted)"
                tunnel_dead = 0
                _restarted_now = True
        except Exception as _e:
            print(f"⚠ トンネル再起動失敗: {type(_e).__name__}: {_e}")

    line = (f"[{time.strftime('%H:%M:%S')}] "
            f"vllm={vllm_st} {TUNNEL_NAME}={cf_st} | GPU: {_gpu()}")

    if url and i % TUNNEL_CHECK_EVERY == 0 and not _restarted_now:
        try:
            with _open_public(f"{url}/healthz", timeout=15) as r:
                line += " | tunnel=OK" if r.status == 200 else f" | tunnel=HTTP{r.status}"
        except Exception as e:
            tunnel_dead += 1
            line += f" | tunnel=DEAD({type(e).__name__}) x{tunnel_dead}"
    _restarted_now = False

    log.append(line)
    print(line)

failed = False
if not ran_any:
    # 即時終了時は、判定材料が無いので現在の死活だけ 1 回確認する。
    # ここで DEAD でも「停止操作による終了」とは区別できないため、
    # 明示的な RUN=0 終了では fail させない（情報は keepalive.txt に残す）。
    vllm_st, cf_st = _proc("vllm serve"), _proc(TUNNEL_NAME)
    print(f"ℹ 即時終了のため死活判定はスキップ（vllm={vllm_st} "
          f"{TUNNEL_NAME}={cf_st}）")
else:
    failed = (vllm_st in ("DEAD", "unknown") or cf_st in ("DEAD", "unknown")
              or tunnel_dead >= 2)
report = "\\n".join([
    f"result={'DEAD' if failed else ('stopped' if not RUN else 'alive')}",
    f"RUN={RUN}",
    f"minutes={KEEPALIVE_MINUTES}",
    f"ran_any={ran_any}",
    f"vllm={vllm_st}",
    f"tunnel={TUNNEL_NAME}:{cf_st}",
    f"url={url}",
    "",
] + log)
saved = save_output("keepalive.txt", report + "\\n")
print()
print("⏱ キープアライブ終了。COMPLETE で /kaggle/working が zip 化されます")

# ---- 次回短縮用: compile / Triton キャッシュを退避 ----
# 成功時のキャッシュを Dataset 化すると次回の起動が短くなる。
# 3GB を超える場合は退避しない（転送と容量の無駄）。
try:
    import tarfile as _tf3
    _keep = []
    for _d in ("/root/.cache/vllm", "/root/.triton", "/root/.cache/triton"):
        if os.path.isdir(_d) and not any(_d.startswith(k + "/") for k in _keep):
            _keep.append(_d)
    _sz = 0
    for _d in _keep:
        for _root, _dirs, _files in os.walk(_d):
            for _f in _files:
                try:
                    _sz += os.path.getsize(os.path.join(_root, _f))
                except OSError:
                    pass
    print(f"🧠 キャッシュ {_sz/1e6:.0f} MB {_keep}")
    if _keep and _sz and _sz < 3e9:
        _out = os.path.join(OUT_DIRS[0], "warm-cache.tar")
        with _tf3.open(_out, "w") as _t:
            for _d in _keep:
                _t.add(_d, arcname=os.path.relpath(_d, "/"))
        print(f"   退避: {_out} {os.path.getsize(_out)/1e6:.0f} MB")
    elif _sz:
        print("   3GB 超のため退避しない")
except Exception as _e:
    print(f"⚠ キャッシュ退避をスキップ: {_e}")

# ============================================================
# 後片付け: GPU / プロセスを解放
# ============================================================
print()
print("=" * 56)
print("🧹 後片付け: GPU / プロセスを解放")
print("=" * 56)


# 順序: 外側（トンネル）→ 中継 → 推論本体（vLLM + EngineCore 子プロセス）
TARGETS = [
    ("-x ngrok", "ngrok"),
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f vllm serve", "vllm"),
    ("-f VLLM::EngineCore", "engine-core"),
]

left = kc_stop_processes(TARGETS, term_wait=5, final_wait=2)

print()
if left:
    print(f"  ⚠ 残存: {', '.join(left)}")
else:
    print("  ✅ 全プロセス停止（vllm / engine-core / tunnel / proxy）")

save_output("cleanup.txt", "\\n".join([
    f"leftover={','.join(left) if left else 'none'}",
    "",
]) + "\\n")
print("  📄 cleanup.txt 保存済")
print("=" * 56)

if failed:
    fail("死活確認に失敗しました（keepalive.txt を確認）", "/tmp/vllm_1.log")
elif not saved:
    fail("keepalive.txt を保存できませんでした")
''')

# ── 書き出し ────────────────────────────────────────────────
pack_nb.write_cells(OUT, cells, "3.10.0")

# ── kernel-metadata.json（push 用・まだ push しない）─────────
SLUG = f"step2-vllm-qwen38-27b-{TUNNEL}"
sources = [DATASET_ID]
if not IS_NGROK:
    sources.append(f"{OWNER}/my-cloudflared-bin")
sources.append(f"{OWNER}/vllm-wheelhouse")
meta = {
    "id": f"{OWNER}/{SLUG}",
    # title は slug に解決できる必要がある（不一致だと push が 409）
    "title": ("Step2 vLLM qwen38 27b cloudflared" if not IS_NGROK
              else SLUG.replace("-", " ")),
    "code_file": OUT.name,
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
meta_path = OUT_DIR / "kernel-metadata.json"
metadata.write(meta_path, meta)
print(f"wrote {meta_path}")
