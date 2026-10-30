#!/usr/bin/env python3
"""Step2 直起動版ビルダー: llama-server を Ollama 経由せず直接起動する。

既存の Ollama 版（build_step2.py）はそのまま残す。こちらは別カーネル。

なぜ直起動か:
  Ollama 0.40.12 は MTP（投機的デコード）のフラグを一切露出していない。
  `ollama serve --help` の環境変数 19 個に spec / mtp / draft が無い。
  llama-server を直に叩けば --spec-type draft-mtp が使える。

代償:
  Ollama が担っていた層を自前で持つ。ただし今回は Dataset に
  model blob と projector blob が既にあるので、pull は不要。

環境変数:
  STEP2D_TUNNEL=ngrok|cloudflared   既定 ngrok
  STEP2D_MTP=1|0                    既定 1（MTP 有効）
  STEP2D_CTX=<int>                  既定 32768（テスト用に小さめ）
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from kbuild import embed, metadata, pack_nb  # noqa: E402
from runtime import source_bundle  # noqa: E402

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
DATASET = "my-ollama-orcarouter-qwen38-27b"
PATCH = "my-ollama-orcarouter-qwen38-27b-patch"

TUNNEL = os.environ.get("STEP2D_TUNNEL", "ngrok").strip().lower()
if TUNNEL not in ("cloudflared", "ngrok"):
    raise SystemExit(f"STEP2D_TUNNEL={TUNNEL!r} は未対応（cloudflared / ngrok）")
IS_NGROK = TUNNEL == "ngrok"

USE_MTP = os.environ.get("STEP2D_MTP", "1").strip() == "1"
CTX = int(os.environ.get("STEP2D_CTX", "32768"))

OUT_DIR = HERE / ("step2-direct-ngrok" if IS_NGROK else "step2-direct")
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / f"step2-direct-{TUNNEL}.ipynb"

PROXY_SRC_B64 = embed.b64_file(HERE / "step2" / "auth_proxy.py")
CHAT_SRC_B64 = embed.b64_file(ROOT / "llama-chat.html")

cells = []

# ── 0: title ────────────────────────────────────────────────
cells.append(("markdown", f"""# 🚀 Step 2 (直起動): llama-server + {TUNNEL}

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

> **Secrets は不要です**（ngrok 版は `NGROK_TOKEN` が必要）。

### このノートの設定

| 項目 | 値 |
|---|---|
| MTP | {'有効（--spec-type draft-mtp）' if USE_MTP else '無効'} |
| コンテキスト | {CTX:,} |
| KV キャッシュ | q8_0 |
| thinking | off |

> ⚠ **テスト用にコンテキストを {CTX:,} に抑えています。** 本番は 131,072。
"""))

# ── 0.5: 共通ランタイム ─────────────────────────────────────
# proxy 起動・トンネル URL 取得・keepalive conf・HTTP 待ちを runtime/ に
# 一本化する。手で編集しない（真実源はリポジトリ直下の runtime/*.py）。
cells.append(("markdown", "---\n### 🧩 共通ランタイム（proxy / tunnel / keepalive / verify）\n\n"
              "`runtime/` の真実源をビルダーが連結して埋め込んでいます。\n"))
cells.append(("code", source_bundle(
    "net", "proc", "storage", "keepalive", "proxy")))

# ── 1: 設定 ─────────────────────────────────────────────────
cells.append(("markdown", "---\n### ⚙️ 設定（GPU検出 → コンテキスト決定）\n"))

cells.append(("code", f'''# ============================================================
# Step 2 (直起動): llama-server を Ollama 経由せず起動
# ============================================================
import json, os, re, subprocess, time

MODEL_NAME = "{MODEL}"
DATASET_SLUG = "{DATASET}"
PATCH_SLUG = "{PATCH}"
USE_MTP = {USE_MTP}
CTX_REQUESTED = {CTX}

# 直起動では Ollama のポートと別にする（既存ノートと衝突させない）
PORT = 11434
PROXY_PORT = 11435

PROXY_SRC_B64 = "{PROXY_SRC_B64}"
CHAT_SRC_B64 = "{CHAT_SRC_B64}"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

LIB_DIR = "/usr/local/lib/ollama"
LLAMA_SERVER = os.path.join(LIB_DIR, "llama-server")

# モデルデータの配置先。既存 Ollama 版のステージング処理をそのまま使うため、
# 同じ変名が必要（_stage がここへ blob/manifest を並べる）。
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"


def _find_input(slug):
    """Dataset のマウント先を探す。

    Kaggle は /kaggle/input/<slug>（旧）と
    /kaggle/input/datasets/<owner>/<slug>（新）が混在する。
    深さ2まで探し、空でない方を採用する。
    """
    cands = [
        f"/kaggle/input/{{slug}}",
        f"/kaggle/input/datasets/yamahei21python/{{slug}}",
    ]
    for base in ("/kaggle/input", "/kaggle/input/datasets"):
        try:
            for owner in os.listdir(base):
                cands.append(os.path.join(base, owner, slug))
        except OSError:
            pass
    for c in cands:
        if os.path.isdir(c) and os.listdir(c):
            return c
    return f"/kaggle/input/{{slug}}"


DATASET_PATH = _find_input(DATASET_SLUG)
PATCH_PATH = _find_input(PATCH_SLUG)

print(f"DATASET : {{DATASET_PATH}}")
print(f"PATCH   : {{PATCH_PATH}}")


def _detect_gpu():
    """全 GPU の (名前, VRAM MiB) を返す。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        if not out:
            return [], []
        names, mems = [], []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            names.append(parts[0].strip())
            mems.append(int(re.sub(r"[^0-9]", "", parts[1])))
        return names, mems
    except Exception as e:
        print(f"GPU 検出失敗: {{e}}")
        return [], []


_gpu_names, _vram_each = _detect_gpu()
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


def _fits(ctx):
    kv_gib = KV_KIB_PER_TOK_Q8 * ctx / 1024 / 1024
    return MODEL_GIB + kv_gib


budget_gib = VRAM_TOTAL / 1024 * 0.88
print(f"予算     : {{budget_gib:.1f}} GiB（実効 VRAM の 88%）")

CONTEXT_LENGTH = CTX_REQUESTED
if _fits(CONTEXT_LENGTH) > budget_gib:
    print(f"⚠ {{CONTEXT_LENGTH:,}} は予算超過 → 段階的に下げる")
    for c in [131072, 65536, 32768, 16384, 8192]:
        if _fits(c) <= budget_gib:
            CONTEXT_LENGTH = c
            break

kv_used = KV_KIB_PER_TOK_Q8 * CONTEXT_LENGTH / 1024 / 1024
print(f"コンテキスト: {{CONTEXT_LENGTH:,}} tokens")
print(f"  モデル {{MODEL_GIB}} GiB + KV(q8_0) {{kv_used:.2f}} GiB = {{MODEL_GIB + kv_used:.2f}} GiB")
print(f"MTP        : {{'有効' if USE_MTP else '無効'}}")

# 失敗時の証跡（FAILED.txt）は共通ランタイム（runtime/storage.py）へ
fail = kc_fail
'''))

# ── 2: バイナリ復元 ─────────────────────────────────────────
cells.append(("markdown", "---\n### 📡 llama-server + CUDA ライブラリ復元（Dataset から）\n"))

cells.append(("code", '''# llama-server と依存 .so を patch Dataset から復元する。
# Ollama 本体（bin/ollama）は使わない。llama-server だけ要る。
import os, shutil, subprocess

MIN_SIZE_LLAMA = 4096   # Linux 版 llama-server は 15 KB の薄いラッパー
# .so の最小サイズ。libggml.so.0 は 220 KB、libggml-base.so.0 は 817 KB と
# 小さいものがある（実測）。1 MB で切ると良品を誤って弾く。
MIN_SIZE_SO = 64 * 1024

LIB_CACHE = os.path.join(PATCH_PATH, "lib", "ollama")

restored, missing = [], []


def _copy(src, dst, min_size):
    """サイズ検証してからコピー。0B/切れファイルで良品を潰さない。"""
    if not (os.path.isfile(src) and os.path.getsize(src) >= min_size):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


# --- llama-server 本体 ---
for cand in (os.path.join(LIB_CACHE, "llama-server"),
             os.path.join(PATCH_PATH, "bin", "llama-server")):
    if _copy(cand, LLAMA_SERVER, MIN_SIZE_LLAMA):
        restored.append("llama-server")
        break
else:
    missing.append("llama-server")

# --- lib/ollama 配下を丸ごと（.so と cuda_v12）---
if os.path.isdir(LIB_CACHE):
    n = 0
    for root, _dirs, files in os.walk(LIB_CACHE):
        rel = os.path.relpath(root, LIB_CACHE)
        dst_dir = LIB_DIR if rel == "." else os.path.join(LIB_DIR, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, fn)
            # llama-server は上で配置済み。上書きしない。
            if os.path.abspath(dst) == os.path.abspath(LLAMA_SERVER):
                continue
            try:
                shutil.copy2(src, dst)
                n += 1
            except OSError as e:
                print(f"  ⚠ copy 失敗: {fn} ({e})")
    restored.append(f"lib/ollama ({n} files)")
else:
    missing.append("lib/ollama")

print(f"✅ 復元: {', '.join(restored) if restored else 'なし'}")
print(f"❌ 不足: {', '.join(missing) if missing else 'なし'}")

# --- 実行ビット ---
# tar 経由の Dataset は実行ビットが落ちる。root 実行では os.access(X_OK) が
# 甘いので mode ビットを直接見る。
for exe in [LLAMA_SERVER]:
    if os.path.isfile(exe) and (os.stat(exe).st_mode & 0o111) != 0o111:
        os.chmod(exe, 0o755)
        print(f"🔧 実行ビット付与: {exe}")

# --- 検証 ---
print()
CUDA_LIB = os.path.join(LIB_DIR, "cuda_v12", "libggml-cuda.so")
IMPL_LIB = os.path.join(LIB_DIR, "libllama-server-impl.so")
ok = True
for path, min_sz, label in [
    (LLAMA_SERVER, MIN_SIZE_LLAMA, "llama-server"),
    (IMPL_LIB, MIN_SIZE_SO, "libllama-server-impl.so"),
    (CUDA_LIB, MIN_SIZE_SO, "cuda_v12/libggml-cuda.so"),
    (os.path.join(LIB_DIR, "libllama.so.0"), MIN_SIZE_SO, "libllama.so.0"),
    (os.path.join(LIB_DIR, "libggml-base.so.0"), MIN_SIZE_SO, "libggml-base.so.0"),
    (os.path.join(LIB_DIR, "libggml.so.0"), MIN_SIZE_SO, "libggml.so.0"),
    (os.path.join(LIB_DIR, "libllama-common.so.0"), MIN_SIZE_SO, "libllama-common.so.0"),
]:
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    mark = "✅" if exists and size >= min_sz else "❌"
    if mark == "❌":
        ok = False
    print(f"{mark} {label}  ({size:,} B)")

if not ok:
    raise RuntimeError(
        "llama-server の依存が揃っていません。patch Dataset を確認してください。"
    )

# --- バージョン確認（実行ビットの end-to-end 証明）---
env = {**os.environ, "LD_LIBRARY_PATH": LIB_DIR + ":" + os.path.join(LIB_DIR, "cuda_v12")}
r = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True, text=True,
                   env=env, timeout=60)
out = (r.stdout or "") + (r.stderr or "")
print()
print(out.strip()[:600])
if r.returncode != 0:
    raise RuntimeError(f"llama-server --version 失敗 (rc={r.returncode})")

# --- プロキシと UI を書き出す ---
import base64 as _b64
with open(PROXY_PATH, "wb") as f:
    f.write(_b64.b64decode(PROXY_SRC_B64))
print(f"✅ auth_proxy.py: {PROXY_PATH} ({os.path.getsize(PROXY_PATH):,} B)")

with open(CHAT_PATH, "wb") as f:
    f.write(_b64.b64decode(CHAT_SRC_B64))
print(f"✅ llama-chat.html: {CHAT_PATH} ({os.path.getsize(CHAT_PATH):,} B)")
'''))

# ── 3: モデル blob 解決 ─────────────────────────────────────
cells.append(("markdown", "---\n### 📦 モデル blob を manifest から解決\n"))

cells.append(("code", '''# Ollama の manifest を読み、model blob と projector blob の実パスを取る。
# Ollama を使わないので pull も Modelfile も不要。Dataset の blob を直に指す。
import json, os

MANIFEST = os.path.join(
    DATASET_PATH, "manifests", "registry.ollama.ai",
    "orcarouter", "Qwen3.8-27B-Uncensored", "latest",
)
BLOBS = os.path.join(DATASET_PATH, "blobs")

print(f"manifest: {MANIFEST}")
if not os.path.isfile(MANIFEST):
    raise RuntimeError(f"manifest が見つかりません: {MANIFEST}")

with open(MANIFEST) as f:
    mf = json.load(f)

MODEL_BLOB = None
MMPROJ_BLOB = None
for layer in mf.get("layers", []):
    mt = layer.get("mediaType", "")
    digest = layer.get("digest", "")
    if not digest.startswith("sha256:"):
        continue
    path = os.path.join(BLOBS, "sha256-" + digest.split(":", 1)[1])
    if mt == "application/vnd.ollama.image.model":
        MODEL_BLOB = path
    elif mt == "application/vnd.ollama.image.projector":
        MMPROJ_BLOB = path

if not MODEL_BLOB or not os.path.isfile(MODEL_BLOB):
    raise RuntimeError(f"model blob が見つかりません: {MODEL_BLOB}")

print(f"✅ model    : {os.path.basename(MODEL_BLOB)}")
print(f"   サイズ   : {os.path.getsize(MODEL_BLOB) / 1024**3:.2f} GiB")

if MMPROJ_BLOB and os.path.isfile(MMPROJ_BLOB):
    print(f"✅ projector: {os.path.basename(MMPROJ_BLOB)}")
    print(f"   サイズ   : {os.path.getsize(MMPROJ_BLOB) / 1024**3:.2f} GiB")
else:
    print("ℹ projector なし（テキストのみ）")
    MMPROJ_BLOB = None
'''))

# ── 4: llama-server 起動 ────────────────────────────────────
cells.append(("markdown", "---\n### 🔥 llama-server 起動（Ollama を経由しない）\n"))

cells.append(("code", '''# モデルデータを配置 → llama-server を直起動する。
#
# ステージング部分は既存の Ollama 版（実証済み）をそのまま流用する。
# 変更点は「ollama serve を起動する」代わりに「llama-server を起動する」だけ。
# これで Dataset のマウント差異・blob 充足・symlink の罠を再発明しない。
import os, shutil, subprocess, time
import glob as _g, json as _json

os.makedirs(OLLAMA_MODELS, exist_ok=True)


def _stage(name, extra_roots=()):
    """DATASET_PATH/name 以下のファイルを実dirとして配置（file symlink）。

    patch Dataset の blobs（config blob sha256-cbe6ad4… 214B）も合流させる。
    これが無いと manifest が要求する blob が欠けて pull 誘発になる。
    """
    dst = os.path.join(OLLAMA_MODELS, name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for root in (DATASET_PATH,) + tuple(extra_roots):
        base = os.path.join(root, name)
        if not os.path.isdir(base):
            continue
        for p in sorted(_g.glob(os.path.join(base, "**"), recursive=True)):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, base)
            d = os.path.join(dst, rel)
            if name == "manifests":
                # manifest は実ファイルのみ正。symlink は ollama が拒否する。
                if os.path.lexists(d) and not os.path.islink(d) and os.path.isfile(d):
                    continue
                if os.path.lexists(d):
                    os.remove(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copyfile(p, d)
                n += 1
                continue
            # blobs: 有効な symlink/実ファイルだけスキップ、壊れていれば作り直す
            if os.path.lexists(d) and os.path.isfile(d):
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                os.symlink(p, d)
            except OSError:
                # 16 GiB を複製すると disk full。巨大ファイルは複製せず落とす。
                if os.path.getsize(p) > (1 << 30):
                    raise
                shutil.copy2(p, d)
            n += 1
    return dst, n


if os.path.isdir(DATASET_PATH) and os.listdir(DATASET_PATH):
    print("📁 Dataset 検出 → モデルデータを準備中...")
    try:
        _, n_blob = _stage("blobs", extra_roots=(PATCH_PATH,))
        _, n_man = _stage("manifests")
    except OSError as _e:
        fail(f"モデルデータ配置に失敗: {type(_e).__name__}: {_e}")
    print(f"✅ blobs {n_blob} 件 / manifests {n_man} 件を配置")

    # manifest が要求する blob の充足を先に点検する。
    _repo, _tag = MODEL_NAME.rsplit(":", 1)
    _man = os.path.join(OLLAMA_MODELS, "manifests", "registry.ollama.ai",
                        *_repo.split("/"), _tag)
    if os.path.isfile(_man):
        with open(_man) as _f:
            _md = _json.load(_f)
        _need = [_md.get("config", {})] + _md.get("layers", [])
        _miss = [e["digest"] for e in _need
                 if e.get("digest") and not os.path.lexists(os.path.join(
                     OLLAMA_MODELS, "blobs",
                     "sha256-" + e["digest"].split(":", 1)[1]))]
        if _miss:
            fail(f"blob 欠落 {len(_miss)} 件: {_miss[:5]}")
        print("✅ manifest 要求 blob 全数充足")
    else:
        fail(f"manifest が見つからない: {_man}")
else:
    fail(f"Dataset が見つかりません: {DATASET_PATH}")

# ---- manifest から model / projector の実パスを取る ----
with open(_man) as _f:
    _mf = _json.load(_f)

MODEL_BLOB = None
MMPROJ_BLOB = None
for _layer in _mf.get("layers", []):
    _mt = _layer.get("mediaType", "")
    _dg = _layer.get("digest", "")
    if not _dg.startswith("sha256:"):
        continue
    _p = os.path.join(OLLAMA_MODELS, "blobs", "sha256-" + _dg.split(":", 1)[1])
    if _mt == "application/vnd.ollama.image.model":
        MODEL_BLOB = _p
    elif _mt == "application/vnd.ollama.image.projector":
        MMPROJ_BLOB = _p

if not MODEL_BLOB or not os.path.isfile(MODEL_BLOB):
    fail(f"model blob が見つかりません: {MODEL_BLOB}")
print(f"✅ model    : {os.path.basename(MODEL_BLOB)} "
      f"({os.path.getsize(MODEL_BLOB) / 1024**3:.2f} GiB)")
if MMPROJ_BLOB and os.path.isfile(MMPROJ_BLOB):
    print(f"✅ projector: {os.path.basename(MMPROJ_BLOB)} "
          f"({os.path.getsize(MMPROJ_BLOB) / 1024**3:.2f} GiB)")
else:
    MMPROJ_BLOB = None
    print("ℹ projector なし（テキストのみ）")

# ---- 既存プロセスの掃除 ----
# 前回セッションの孤児 llama-server が残っているとポート/VRAM を奪う。
subprocess.run(["pkill", "-TERM", "-f", "llama-server"], capture_output=True)
time.sleep(3)
subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
time.sleep(2)

# ---- llama-server 起動 ----
# LD_LIBRARY_PATH が無いと libllama-server-impl.so を dlopen できず、
# エラーを出さずに CPU 推論へ落ちる（実測 0.8 tps）。
LIB_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": ":".join([LIB_DIR, os.path.join(LIB_DIR, "cuda_v12")]),
}

args = [
    LLAMA_SERVER,
    "-m", MODEL_BLOB,
    "-c", str(CONTEXT_LENGTH),
    # KV は q8_0 を明示。f16 既定だと長い ctx で VRAM を溢れて CPU に落ちる。
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "-ngl", "all",
    "-fa", "on",
    "-np", "1",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--jinja",
    "--temp", "0",
    "--alias", MODEL_NAME,
    # thinking を切る。切らないと content が空で reasoning だけ返る。
    "--reasoning", "off",
    "--reasoning-budget", "0",
    "--cors-origins", "*",
]

# MTP（投機的デコード）。--mtp は download 専用、--nextn は imatrix 専用。
# サーバーで使う正しいフラグは --spec-type draft-mtp。
if USE_MTP:
    args += ["--spec-type", "draft-mtp"]

if MMPROJ_BLOB:
    args += ["--mmproj", MMPROJ_BLOB]

print()
print("起動コマンド:")
print("  " + " ".join(args))
print()

LOG = "/tmp/llama-server.log"
with open(LOG, "w") as lf:
    proc = subprocess.Popen(args, env=LIB_ENV, stdout=lf, stderr=lf)
print(f"🔄 llama-server PID: {proc.pid}")

# ---- 起動待ち ----
# /health が 200 を返すまで。モデルロード込みで 5〜10 分かかる。
# OLLAMA_LOAD_TIMEOUT=600 相当を待つ（既存版と同じ 10 分 + 余裕）。
print("⏳ ロード待ち（最大 900s）...")
ready = False
t0 = time.time()
for i in range(180):
    if proc.poll() is not None:
        print(open(LOG).read()[-5000:])
        fail(f"llama-server が即終了しました (rc={proc.returncode})", LOG)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        pass
    if i % 12 == 11:
        print(f"   ... {int(time.time() - t0)}s")
    time.sleep(5)

if not ready:
    print(open(LOG).read()[-6000:])
    fail("llama-server が 900s 以内に応答しませんでした", LOG)

print(f"✅ 起動完了（{int(time.time() - t0)}s）")

# ---- GPU に載っているか（CPU 落ちの検出）----
log_txt = open(LOG).read()
if "no usable GPU found" in log_txt:
    fail("GPU が認識されていません（CPU 推論に落ちています）", LOG)
if "using CPU" in log_txt:
    print("⚠ 起動ログに 'using CPU' があります")

print()
print("--- 起動ログの要点 ---")
for line in log_txt.splitlines():
    if any(k in line for k in ["CUDA", "offloaded", "MTP", "spec", "draft",
                               "n_ctx", "model size", "using device"]):
        print("  " + line.strip()[:130])
'''))

# ── 5: プロキシ起動 ─────────────────────────────────────────
cells.append(("markdown", "---\n### 🌐 リバースプロキシ起動（UI 配信 + 同一オリジン化）\n"))

cells.append(("code", '''# プロキシを起動。UI を / で配り、API を llama-server へ中継する。
# 起動・待機・認証なし検査の実装は共通ランタイム（runtime/proxy.py）。
import os

proxy_env = {**os.environ, "API_USER": "", "API_PASS": ""}
proxy_proc = kc_start_proxy(
    PROXY_PORT, f"http://127.0.0.1:{PORT}", CHAT_PATH, PROXY_PATH,
    env=proxy_env, fail=fail,
)

# 認証は付けない設計。401 が返るなら設定ミス。
kc_assert_no_auth(PROXY_PORT, fail=fail)
'''))

# ── 6: 疎通確認（MTP の効果測定込み）────────────────────────
cells.append(("markdown", "---\n### ✅ 疎通確認 + 速度測定\n"))

cells.append(("code", '''# 直起動が動いているか、MTP が効いているかを実測する。
import json, time, urllib.request

BASE = f"http://127.0.0.1:{PROXY_PORT}"


def _post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


print("[1] /v1/models")
models = _get("/v1/models")
ids = [m.get("id") for m in models.get("data", [])]
print(f"    ✅ {ids}")

print("[2] 短文応答")
t0 = time.time()
resp = _post("/v1/chat/completions", {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "stream": False,
    "max_tokens": 16,
    "temperature": 0,
})
dt = time.time() - t0
msg = resp.get("choices", [{}])[0].get("message", {})
print(f"    ✅ {dt:.1f}s  reply={msg.get('content', '')!r}")

print("[3] 速度測定（128 tokens 生成）")
# 同じプロンプトで 2 回測る。1 回目はプロンプト処理込み、2 回目が実効速度。
PROMPT = "Write a short paragraph about the ocean."
rates = []
for run in range(2):
    t0 = time.time()
    resp = _post("/v1/chat/completions", {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "max_tokens": 128,
        "temperature": 0,
    })
    dt = time.time() - t0
    usage = resp.get("usage", {})
    ct = usage.get("completion_tokens", 0)
    rate = ct / dt if dt > 0 else 0
    rates.append(rate)
    print(f"    run{run+1}: {ct} tokens / {dt:.2f}s = {rate:.1f} t/s")

print()
print("=" * 56)
print(f"  MTP      : {'有効（--spec-type draft-mtp）' if USE_MTP else '無効'}")
print(f"  ctx      : {CONTEXT_LENGTH:,}")
print(f"  実効速度 : {rates[-1]:.1f} t/s")
print("=" * 56)

# 比較用に結果を保存
result = {
    "mtp": USE_MTP,
    "context": CONTEXT_LENGTH,
    "rates": rates,
    "effective_tps": rates[-1],
}
with open("/kaggle/working/bench_direct.json", "w") as f:
    json.dump(result, f, indent=2)
print("📄 bench_direct.json 保存済")
'''))

# ── 7: トンネル ─────────────────────────────────────────────
cells.append(("markdown", f"---\n### 🔗 {TUNNEL} トンネル → 公開URL\n"))

if IS_NGROK:
    cells.append(("code", '''# ngrok 固定ドメインで公開する。
import json, os, subprocess, time, urllib.request

NGROK_BIN = "/usr/local/bin/ngrok"
NGROK_DOMAIN_DEFAULT = "rebalance-cornflake-sagging.ngrok-free.dev"

if not os.path.isfile(NGROK_BIN):
    subprocess.run(
        ["bash", "-c",
         "curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
         " | tar xz -C /usr/local/bin"],
        check=True)

# authtoken は Secrets から
_tok = os.environ.get("NGROK_TOKEN", "")
if not _tok:
    try:
        from kaggle_secrets import UserSecretsClient
        _tok = UserSecretsClient().get_secret("NGROK_TOKEN") or ""
    except Exception:
        _tok = ""
if not _tok:
    raise RuntimeError("NGROK_TOKEN がありません（Kaggle Secrets に登録してください）")
subprocess.run([NGROK_BIN, "config", "add-authtoken", _tok],
               check=True, capture_output=True)

NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip()
if not NGROK_DOMAIN:
    try:
        from kaggle_secrets import UserSecretsClient
        NGROK_DOMAIN = (UserSecretsClient().get_secret("NGROK_DOMAIN") or "").strip()
    except Exception:
        NGROK_DOMAIN = ""
if not NGROK_DOMAIN:
    NGROK_DOMAIN = NGROK_DOMAIN_DEFAULT
print("📌 固定ドメイン: " + NGROK_DOMAIN)

# --pooling-enabled が無いと、同じドメインが残っているとき
# ERR_NGROK_334 で即死する（実測）。--domain は deprecated、--url が現行。
RESTART_CMD = [NGROK_BIN, "http", str(PROXY_PORT),
               "--log=stdout", "--pooling-enabled",
               "--url=https://" + NGROK_DOMAIN]
TUNNEL_LOG = "/tmp/ngrok.log"

subprocess.run(["pkill", "-x", "ngrok"], capture_output=True)
time.sleep(1)
with open(TUNNEL_LOG, "w") as _log:
    ng_proc = subprocess.Popen(RESTART_CMD, stdout=_log, stderr=_log)
print("🔄 ngrok PID: " + str(ng_proc.pid))

public_url = kc_wait_url("ngrok", TUNNEL_LOG, progress_every=0)

if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    raise RuntimeError("ngrok の公開URL取得に失敗しました")

print("🌍 URL: " + public_url)
with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
print("=" * 60)
print("  🌍 PUBLIC URL : " + public_url)
print("=" * 60)
'''))
else:
    cells.append(("code", '''# cloudflared quick tunnel で公開する（アカウント不要）。
import os, re, subprocess, time

CF_BIN = "/usr/local/bin/cloudflared"
CF_PATH = None
for cand in (os.path.join(PATCH_PATH, "bin", "cloudflared"),
             "/usr/local/bin/cloudflared"):
    if os.path.isfile(cand):
        CF_PATH = cand
        break

if CF_PATH is None:
    subprocess.run(
        ["bash", "-c",
         "curl -fsSL -o /usr/local/bin/cloudflared "
         "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
         " && chmod +x /usr/local/bin/cloudflared"],
        check=True)
    CF_PATH = "/usr/local/bin/cloudflared"

RESTART_CMD = [CF_PATH, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]
TUNNEL_LOG = "/tmp/cloudflared.log"

subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
time.sleep(1)
with open(TUNNEL_LOG, "w") as _log:
    cf_proc = subprocess.Popen(RESTART_CMD, stdout=_log, stderr=_log)
print("🔄 cloudflared PID: " + str(cf_proc.pid))

public_url = kc_wait_url("cloudflared", TUNNEL_LOG, progress_every=0)

if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    raise RuntimeError("cloudflared の公開URL取得に失敗しました")

print("🌍 URL: " + public_url)
with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
print("=" * 60)
print("  🌍 PUBLIC URL : " + public_url)
print("=" * 60)
'''))

# ── 8: 接続情報 ─────────────────────────────────────────────
cells.append(("markdown", "---\n### 📋 接続情報\n"))

cells.append(("code", '''# 接続情報を保存する（セル出力は後から取れないため）。
import json, os

try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    raise RuntimeError("公開URLがありません。トンネルセルを実行してください。")

opencode_cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "kaggle-direct": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Kaggle llama-server (direct)",
            "options": {"baseURL": f"{url}/v1", "apiKey": "dummy"},
            "models": {MODEL_NAME: {"name": "Qwen3.8 27B (direct + MTP)"}},
        }
    },
    "model": f"kaggle-direct/{MODEL_NAME}",
}

buf = [
    "=" * 64,
    f"  🌍 Public URL : {url}",
    f"  💬 チャット UI : {url}/",
    f"  🤖 Model      : {MODEL_NAME}",
    f"  📏 Context    : {CONTEXT_LENGTH:,}",
    f"  ⚡ MTP        : {'有効' if USE_MTP else '無効'}",
    "  🔓 認証        : なし",
    "=" * 64,
    "",
    "📋 curl:",
    f"  curl {url}/api/tags",
    "",
    "📋 opencode.json:",
    json.dumps(opencode_cfg, indent=2, ensure_ascii=False),
    "",
    "⚠ 直起動版: Ollama を使っていません。llama-server が直接応答します",
    "⚠ Kaggle: 9時間でカーネルが停止します",
]
report = "\\n".join(buf)
print(report)

with open("/kaggle/working/connection_info.txt", "w") as f:
    f.write(report + "\\n")
print("📄 connection_info.txt 保存済")
'''))

# ── 9: keepalive ────────────────────────────────────────────
cells.append(("markdown", """---
### 🔋 キープアライブ

既定 **3 分**（テスト用）。`KEEPALIVE_MINUTES` を変えれば延ばせます（上限 480）。

- チャット送信で残り時間がリセットされます（`/tmp/keepalive.conf`）
- 「■ 停止」ボタンで `RUN=0` が書かれ、ループが終了します
"""))

cells.append(("code", '''# キープアライブ: 有限ループ。テスト用に短くしてある。
import json, os, subprocess, time, urllib.request

KEEPALIVE_CONF = "/tmp/keepalive.conf"
KEEPALIVE_MINUTES = 3      # テスト用。本番は 480。
MAX_MINUTES = 480


def _read_conf():
    """RUN=1/0 と MINUTES を読む（既定値・上限は共通ランタイム）。"""
    run, mins, _mtime = kc_read_conf(
        KEEPALIVE_CONF, KEEPALIVE_MINUTES, MAX_MINUTES,
        on_invalid="zero", quiet_clamp=True)
    return run, mins


# 初期値を書く（プロキシが送信時に上書きする）
kc_write_conf(KEEPALIVE_CONF, 1, KEEPALIVE_MINUTES)

try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    url = ""

RUN, mins = _read_conf()
print(f"Public URL: {url}")
if RUN:
    print(f"🔋 ON。{mins}分間維持（チャット送信でリセット、■ で停止）")
else:
    print("🛑 OFF（RUN=0）。即時終了します")

log = []
for i in range(1, mins + 1):
    RUN, mins = _read_conf()
    if not RUN:
        print(f"RUN=0 で {i}周目にキープアライブ終了")
        break
    time.sleep(60)

    # プロセスの死活
    def _alive(pattern):
        return kc_proc_state(pattern, "argv")

    ls = _alive("llama-server")
    px = _alive("auth_proxy.py")
    tn = _alive("ngrok" if "ngrok" in url else "cloudflared")

    # GPU 使用量
    gpu = kc_gpu_usage()

    line = f"[{time.strftime('%H:%M:%S')}] llama={ls} proxy={px} tunnel={tn} | GPU: {gpu}"
    log.append(line)
    print(line)

# 証跡
report = "\\n".join([
    f"result={'stopped' if not RUN else 'alive'}",
    f"minutes={mins}",
    f"url={url}",
    "",
] + log)
with open("/kaggle/working/keepalive.txt", "w") as f:
    f.write(report + "\\n")
print()
print("⏱ キープアライブ終了")
print("📄 keepalive.txt 保存済")

# --- 後片付け: GPU を明示的に手放す ---
print()
print("=" * 56)
print("🧹 後片付け: GPU / プロセスを解放")
print("=" * 56)

TARGETS = [
    ("-x ngrok", "ngrok"),
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f llama-server", "llama-server"),
]

left = kc_stop_processes(TARGETS)

if left:
    print(f"  ⚠ 残存: {', '.join(left)}")
else:
    print("  ✅ 全プロセス停止")

with open("/kaggle/working/cleanup.txt", "w") as f:
    f.write(f"leftover={','.join(left) if left else 'none'}\\n")
print("📄 cleanup.txt 保存済")
'''))

# ── 書き出し ────────────────────────────────────────────────
pack_nb.write(OUT, cells)

# kernel-metadata.json
meta = {
    "id": f"yamahei21python/step2-direct-llama-server-{TUNNEL}",
    "title": f"Step2 direct llama-server {TUNNEL}",
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