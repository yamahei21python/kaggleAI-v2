#!/usr/bin/env python3
"""Step2 notebook builder: orcarouter/Qwen3.8-27B-Uncensored + Ollama on Kaggle T4 x2

トンネルは環境変数で切り替える（セル構成は共通、トンネルセルだけ差し替え）:
    STEP2_TUNNEL=cloudflared  (既定) → アカウント不要・URL は毎回ランダム
    STEP2_TUNNEL=ngrok                → NGROK_TOKEN 必須・NGROK_DOMAIN で固定URL可
"""
import base64
import json
import os
import pathlib

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
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
pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)

# Basic 認証つきリバースプロキシを notebook に埋め込む（base64）。
# 単一のソース（kaggle/step2/auth_proxy.py）をそのまま使うので、
# notebook 側でコードを二重管理しない。
PROXY_SRC_B64 = base64.b64encode(
    (HERE / "step2" / "auth_proxy.py").read_bytes()
).decode()

# チャット UI。proxy が / で配信する。
# ローカル版（リポジトリ直下の llama-chat.html）をそのまま使う。
# API の接続先は HTML 側が自動判定する（?api= → ?port= → 同一オリジン）
# ので、ローカルでも Kaggle 中継でも同じファイルが動く。
CHAT_SRC_B64 = base64.b64encode(
    (HERE.parent / "llama-chat.html").read_bytes()
).decode()

cells = []

# ── 0: title ────────────────────────────────────────────────
cf_bullet = ("" if IS_NGROK else
             "   - `my-cloudflared-bin`（cloudflared 40 MB）— 無いと毎回 GitHub から落とす")
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

cells.append(("markdown", f"""# 🚀 Step 2: Ollama + {TUNNEL} 公開APIサーバー

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
"""))

# ── 1: 設定 ─────────────────────────────────────────────────
cells.append(("markdown", "---\n### ⚙️ 設定（GPU検出 → モデル・コンテキスト決定）\n"))

cells.append(("code", f'''# ============================================================
# Step 2: Ollama + cloudflared 公開APIサーバー
# ============================================================
import os, re, json, time, subprocess, glob as _glob


def _find_input(slug):
    \"\"\"Dataset のマウント場所を探す。

    Kaggle はマウント形式を従来の /kaggle/input/<slug> から
    /kaggle/input/datasets/<owner>/<slug> へ移行中（同一日内に両方を実測）。
    深さ2まで探索し、最初に「空でない」ディレクトリを返す。
    \"\"\"
    cands = []
    # 文字列連結にする（このセルは f-string で生成されるため、
    # バレース表記で書くと生成物にリテラルが残り、glob が
    # プレースホルダ文字列を探す事故になる）
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


MODEL_NAME = "{MODEL}"
DATASET_PATH = _find_input("{DATASET}")
PATCH_PATH = _find_input("{PATCH}")
# cloudflared 専用 Dataset。フォルダ構造は flatten されることがあるので
# インストール側で bin/ の有無両方を探索する。
CF_PATH = _find_input("my-cloudflared-bin")
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
PORT = 11434

# Basic 認証プロキシのソース（base64）。ビルダーが auth_proxy.py を埋め込む。
PROXY_SRC_B64 = "{PROXY_SRC_B64}"
# チャット UI（proxy が / で返す）
CHAT_SRC_B64 = "{CHAT_SRC_B64}"
CHAT_PATH = "/usr/local/bin/llama-chat.html"


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
VRAM_TOTAL = sum(_vram_each)          # ← 合計で判定する（per-GPU ではない）
print(f"GPU      : {{_gpu_names}}")
print(f"VRAM     : {{_vram_each}} MiB each  →  合計 {{VRAM_TOTAL}} MiB")

if VRAM_TOTAL == 0:
    raise RuntimeError("GPU が見つかりません。Accelerator を 'GPU T4 x2' に設定してください。")

# ---- コンテキスト長を VRAM から決める ----
# 実測: full attention 17層 × kv_head 4 × (key 256 + value 256) × 2B = 68 KiB/token (f16)
#       q8_0 KV ならその半分 = 34 KiB/token
MODEL_GIB = 16.52
KV_KIB_PER_TOK_Q8 = 34.0

def _fits(ctx):
    kv_gib = KV_KIB_PER_TOK_Q8 * ctx / 1024 / 1024
    return MODEL_GIB + kv_gib

# 実効 VRAM は 30 GiB ではなく余裕を見て 0.88 倍で見積もる
budget_gib = VRAM_TOTAL / 1024 * 0.88
print(f"予算     : {{budget_gib:.1f}} GiB（実効 VRAM の 88%）")

# 実測（kaggle/bench/）: 32K=13.7 tps / 131K=13.7 tps / 262K=3.7 tps。
# 131K までは速度が落ちない。262K で 1/3.7 になるので既定は 131K。
CTX_CANDIDATES = [131072, 65536, 32768, 16384, 8192]
CONTEXT_LENGTH = 8192
for c in CTX_CANDIDATES:
    if _fits(c) <= budget_gib:
        CONTEXT_LENGTH = c
        break

kv_used = KV_KIB_PER_TOK_Q8 * CONTEXT_LENGTH / 1024 / 1024
print(f"コンテキスト: {{CONTEXT_LENGTH:,}} tokens")
print(f"  モデル {{MODEL_GIB}} GiB + KV(q8_0) {{kv_used:.2f}} GiB = {{MODEL_GIB + kv_used:.2f}} GiB")

# ---- 公開ポート ----
# cloudflared の quick tunnel には Basic 認証が無い。ここでは認証を付けない
# （UI を開いただけで使えることを優先）。URL を知れば誰でも使える点は許容する。
# 認証を戻したい場合は環境変数 API_PASS を設定すれば proxy が有効化する。
API_USER = os.environ.get("API_USER", "") or "ollama"
API_PASS = os.environ.get("API_PASS", "")

# 公開ポート: プロキシが待ち受ける。Ollama 本体は 11434 のまま。
PROXY_PORT = 11435

# トンネル種別（ビルダーが埋める）。cloudflared 前提の処理を分岐するために使う。
TUNNEL_KIND = "{TUNNEL}"

print()
print("=" * 56)
print(f"  認証        : {{'OFF（URL を知れば誰でも利用可）' if not API_PASS else 'ON'}}")
print(f"  公開ポート  : {{PROXY_PORT}}（Ollama 本体は {{PORT}}）")
print("=" * 56)
print("  Secrets の設定は不要です（cloudflared はアカウント不要）")

# ---- 検証成果物の永続化 ----
# `kaggle kernels output` が zip 化するのは **/kaggle/working** のだけ。
# /output は公式の永続先ではなく、作れても zip に載らないことがある
#（= 保存したつもりで証跡ゼロ）。だから両方へ書く。
OUT_DIRS = ("/kaggle/working", "/output")
PROXY_PATH = "/usr/local/bin/auth_proxy.py"


def save_output(name, text):
    """検証成果物を永続領域へ。1 つでも書ければ True（成功条件に使う）。"""
    saved = []
    for d in OUT_DIRS:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
            saved.append(d)
        except Exception as e:
            print(f"  ⚠ {{d}} 書き出し失敗 ({{e}})")
    if saved:
        print(f"  📄 {{name}} 保存済: {{', '.join(saved)}}")
        return True
    print(f"  ⚠ 成果物を1つも保存できませんでした: {{name}}")
    return False


def fail(msg, tail_path=None):
    """失敗原因を永続領域に残してから raise。

    ログ API は実行中も空・完了後も非公開なので、**証跡が唯一の手掛かり**。
    raise だけだと原因が分からず、COMPLETE と ERROR を突き合わせられない。
    """
    tail = ""
    try:
        if tail_path and os.path.isfile(tail_path):
            tail = open(tail_path).read()[-3000:]
    except Exception:
        pass
    save_output("FAILED.txt", f"{{msg}}\\n\\n{{tail}}")
    raise RuntimeError(msg)
'''))

# ── 2: install ──────────────────────────────────────────────
cells.append(("markdown", "---\n### 📡 Ollama / cloudflared インストール（Dataset キャッシュ優先）\n"))

cells.append(("code", '''# バイナリ復元: Dataset キャッシュ優先、不足分のみ DL
import os, shutil, subprocess

BIN_DIR = "/usr/local/bin"
LIB_DIR = "/usr/local/lib/ollama"
BIN_CACHE = os.path.join(PATCH_PATH, "bin")
LIB_CACHE = os.path.join(PATCH_PATH, "lib", "ollama")
MIN_SIZE = 1024 * 1024  # 1MB 未満は壊れているとみなす
# llama-server は例外。Linux 版は 15 KB の薄いラッパーで、
# 実体は libllama-server-impl.so を dlopen する（1MB 判定だと誤って弾く）
MIN_SIZE_LLAMA = 4096

LLAMA_SERVER_DST = os.path.join(LIB_DIR, "llama-server")
# install.sh は llama-server を lib/ollama/ に置く。patch Dataset もその配置で
# 作ってあるので lib 側を第一候補、旧配置の bin/ 側も後方互換で見る。
# サイズ検証を先に行う: 0B/切れファイルが良品を shadow すると missing 判定になり
# install.sh が走って ollama が上書きされる（version 固定が崩れる）。
LLAMA_SERVER_CACHE = None
for _c in (os.path.join(LIB_CACHE, "llama-server"),
           os.path.join(BIN_CACHE, "llama-server")):
    if os.path.isfile(_c) and os.path.getsize(_c) >= MIN_SIZE_LLAMA:
        LLAMA_SERVER_CACHE = _c
        break

restored, missing = [], []


def _find_cached(name):
    """バイナリのキャッシュ位置を探す。見つからなければ None。

    cloudflared は専用 Dataset（my-cloudflared-bin）から復元する。
    Dataset 側のフォルダ構造は flatten されることがあるので、
    `bin/` あり / なし の両方を試してから、既存の patch Dataset に流す。
    """
    candidates = [os.path.join(BIN_CACHE, name)]
    if name == "cloudflared":
        candidates = [
            os.path.join(CF_PATH, "cloudflared"),
            os.path.join(CF_PATH, "bin", "cloudflared"),
        ] + candidates
    for src in candidates:
        if os.path.isfile(src) and os.path.getsize(src) >= MIN_SIZE:
            return src
    return None


# --- ollama / cloudflared ---
for name in (["ollama"] if TUNNEL_KIND == "ngrok" else ["ollama", "cloudflared"]):
    dst = os.path.join(BIN_DIR, name)
    src = _find_cached(name)
    if src:
        shutil.copy2(src, dst)
        subprocess.run(["chmod", "+x", dst])
        restored.append(f"{name} <- {src}")
    else:
        missing.append(name)

# --- llama-server（Ollama 0.5+ で必須）---
if LLAMA_SERVER_CACHE and os.path.getsize(LLAMA_SERVER_CACHE) >= MIN_SIZE_LLAMA:
    os.makedirs(LIB_DIR, exist_ok=True)
    shutil.copy2(LLAMA_SERVER_CACHE, LLAMA_SERVER_DST)
    subprocess.run(["chmod", "+x", LLAMA_SERVER_DST])
    restored.append("llama-server")
else:
    missing.append("llama-server")

# --- lib/ollama（CUDA バックエンド）---
# これが無いと llama-server が libggml-cuda.so を dlopen できず、
# エラーを出さずに CPU 推論へ落ちる（実測 0.9 tps / VRAM 0 MiB）。
if os.path.isdir(LIB_CACHE):
    os.makedirs(LIB_DIR, exist_ok=True)
    n_lib = 0
    for root, _dirs, files in os.walk(LIB_CACHE):
        rel = os.path.relpath(root, LIB_CACHE)
        dst_dir = LIB_DIR if rel == "." else os.path.join(LIB_DIR, rel)
        os.makedirs(dst_dir, exist_ok=True)
        for fn in files:
            src_p = os.path.join(root, fn)
            dst_p = os.path.join(dst_dir, fn)
            shutil.copy2(src_p, dst_p)
            # copy2 は mode を保存する。tar 経由の Dataset では x ビットが全部
            # 落ちる（v13 実測 → fork/exec permission denied）。root 実行の
            # os.access(X_OK) は「x ビットが1つでもあれば True」なので、全部
            # 落ちたケースを判定できない。lib/ollama の中身は .so と
            # llama-server だけ＝exec ビット過剰でも無害なので無条件 0755。
            os.chmod(dst_p, 0o755)
            n_lib += 1
    restored.append(f"lib/ollama ({n_lib} files)")
else:
    missing.append("lib/ollama")

# lib walk が llama-server を上書きするので、実行ビットは walk の後に確定させる
if os.path.isfile(LLAMA_SERVER_DST):
    os.chmod(LLAMA_SERVER_DST, 0o755)

# CUDA バックエンドの実在確認（これが GPU 推論の前提）
CUDA_LIB = os.path.join(LIB_DIR, "cuda_v12", "libggml-cuda.so")
print(f"{'✅' if os.path.isfile(CUDA_LIB) else '⚠ '} CUDA バックエンド: {CUDA_LIB}")

print(f"✅ キャッシュ復元: {', '.join(restored) if restored else 'なし'}")
print(f"⬇️  DL が必要  : {', '.join(missing) if missing else 'なし'}")

# --- 不足分だけダウンロード ---
# ollama 本体が在っても llama-server / CUDA が欠けている限り install.sh は必要
# （patch 側に llama-server が無いため v8 で欠落を実測。無いと検証で即 fail）。
# install.sh が ollama を上書きする点は許容 — 上書きが機能不全になる风险より、
# llama-server 欠落で落ちる方を避ける。patch に llama-server 入れるのが本筋。
if missing:
    need_ollama = ("ollama" in missing) or ("llama-server" in missing) or ("lib/ollama" in missing)
    procs = []

    if need_ollama:
        # zstd は ollama の .tar.zst 展開に必要
        if subprocess.run(["dpkg", "-s", "zstd"], capture_output=True).returncode != 0:
            subprocess.run("apt-get install -y -qq zstd", shell=True, check=False)
        print("⬇️  ollama をインストール中...")
        procs.append(subprocess.Popen(
            "curl -fsSL --connect-timeout 15 https://ollama.com/install.sh | sh",
            shell=True,
        ))

    if "cloudflared" in missing:
        print("⬇️  cloudflared をダウンロード中...")
        procs.append(subprocess.Popen(
            "curl -fsSL --connect-timeout 15 -o /usr/local/bin/cloudflared "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && "
            "chmod +x /usr/local/bin/cloudflared",
            shell=True,
        ))

    for p in procs:
        # 無限待ちはハングで notebook が止まり成果物が残らない。
        # install.sh（tar.zst 展開込み）・curl とも最大10分まで。
        try:
            p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            fail("バイナリ取得がタイムアウトしました（10分）")

# --- 検証 ---
print()
ok = True
_targets = [(f"{BIN_DIR}/ollama", MIN_SIZE),
            (LLAMA_SERVER_DST, MIN_SIZE_LLAMA),
            (CUDA_LIB, MIN_SIZE)]
if TUNNEL_KIND != "ngrok":
    _targets.insert(1, (f"{BIN_DIR}/cloudflared", MIN_SIZE))
for path, min_sz in _targets:
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    mark = "✅" if exists and size >= min_sz else "❌"
    if mark == "❌":
        ok = False
    print(f"{mark} {path}  ({size:,} B)")

# 実行ビットは tar 経由の Dataset で落ちる（v13 実測: fork/exec permission denied）。
# サイズだけ見ていると通ってしまい、起動時に 500 になるので明示的に確認する。
# root 実行では os.access(X_OK) が「x ビットが1つでもあれば True」になるため、
# mode ビットで 3 桁すべて（0o111）が立っているかを見る。
_exes = [f"{BIN_DIR}/ollama", LLAMA_SERVER_DST]
if TUNNEL_KIND != "ngrok":
    _exes.insert(1, f"{BIN_DIR}/cloudflared")
for exe in _exes:
    if not os.path.isfile(exe):
        continue
    if os.stat(exe).st_mode & 0o111 != 0o111:
        try:
            os.chmod(exe, 0o755)
            print(f"🔧 実行ビット付与: {exe}")
        except OSError as e:
            print(f"❌ chmod 失敗: {exe} ({e})")
            ok = False
    if os.stat(exe).st_mode & 0o111 != 0o111:  # 事後検証（noexec 等を検出）
        print(f"❌ 実行ビット未設定: {exe}")
        ok = False

if not ok:
    fail("バイナリの準備に失敗しました（cloudflared は Dataset から、"
         "それ以外は GitHub / ollama.com から取る想定）")

print()
# --version が通る = 実行ビットの end-to-end 証明。returncode まで見る
# （exec 失敗だと stdout が空のまま素通りする）。
_ver_exes = [f"{BIN_DIR}/ollama"]
if TUNNEL_KIND != "ngrok":
    _ver_exes.append(f"{BIN_DIR}/cloudflared")
for _exe in _ver_exes:
    _r = subprocess.run([_exe, "--version"], capture_output=True, text=True)
    print((_r.stdout or _r.stderr).strip())
    if _r.returncode != 0:
        fail(f"{_exe} 起動失敗 (returncode={_r.returncode})")

# --- Basic 認証プロキシを書き出す（base64 埋め込み）---
# PROXY_PATH は設定セルの定義を使う（二重管理しない）
import base64 as _b64
with open(PROXY_PATH, "wb") as f:
    f.write(_b64.b64decode(PROXY_SRC_B64))
print(f"✅ auth_proxy.py を書き出し: {PROXY_PATH} "
      f"({os.path.getsize(PROXY_PATH):,} B)")

with open(CHAT_PATH, "wb") as f:
    f.write(_b64.b64decode(CHAT_SRC_B64))
print(f"✅ llama-chat.html を書き出し: {CHAT_PATH} "
      f"({os.path.getsize(CHAT_PATH):,} B)")
'''))

# ── 3: auth proxy ───────────────────────────────────────────
cells.append(("markdown", "---\n### 🌐 リバースプロキシ起動（UI 配信 + 同一オリジン化）\n"))

cells.append(("code", '''# リバースプロキシを起動（UI を / で配信し、API を同一オリジンにする）
# 認証は既定で無効（環境変数 API_PASS を設定したときだけ有効）
# import os は単体実行でも NameError にならないように（設定セル非依存に）
import os, subprocess, time, urllib.request, urllib.error

# 前回のプロキシを落とす
subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
time.sleep(1)

proxy_env = {
    **os.environ,
    # 認証なし（proxy は API_PASS が空なら素通しする）
    "API_USER": "",
    "API_PASS": "",
}
with open("/tmp/auth_proxy.log", "w") as log:
    proxy_proc = subprocess.Popen(
        ["python3", PROXY_PATH,
         "--port", str(PROXY_PORT),
         "--upstream", f"http://127.0.0.1:{PORT}",
         "--chat-html", CHAT_PATH],
        env=proxy_env, stdout=log, stderr=log,
    )
print(f"🔄 proxy PID: {proxy_proc.pid}")

# 起動待機（/healthz は上流を見ないので即応答する）
for i in range(30):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PROXY_PORT}/healthz", timeout=3
        ) as r:
            if r.status == 200:
                print(f"✅ プロキシ起動完了（{i+1}s）")
                break
    except Exception:
        pass
    time.sleep(1)
else:
    print(open("/tmp/auth_proxy.log").read()[-2000:])
    fail("認証プロキシの起動に失敗しました", "/tmp/auth_proxy.log")

# 認証は付けない（設計）。401 が返るなら設定ミス。
# 200 は Ollama 起動前だと 502 になるため、ここでは 401/403 でないことだけ見る。
# 完全判定（200 / streaming / OpenAI互換）は公開URL経由の verify セルへ一本化。
req = urllib.request.Request(f"http://127.0.0.1:{PROXY_PORT}/api/tags")
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"  ✅ 認証なし → HTTP {r.status}（401 でない）")
except urllib.error.HTTPError as e:
    if e.code in (401, 403):
        fail(f"認証なしで HTTP {e.code} — 認証が有効になっています")
    else:
        print(f"  ✅ 認証なし → HTTP {e.code}（上流未起動。401 でない）")
except Exception as e:
    fail(f"プロキシ検査に失敗: {type(e).__name__}: {e}")
'''))

# ── 4: ollama serve ─────────────────────────────────────────
cells.append(("markdown", "---\n### 🔥 Ollama サーバー起動\n"))

cells.append(("code", '''# Dataset を展開 → Ollama 起動（コンテキスト長・KV 量子化を明示）
import os, shutil, subprocess, time

os.makedirs(OLLAMA_MODELS, exist_ok=True)

# --- モデルデータの準備 ---
# blobs は「ディレクトリごと symlink」ではなく「実ディレクトリ + ファイル単位
# symlink」にする。dir ごと symlink だと OLLAMA_MODELS 配下が read-only
# （/kaggle/input は読み取り専用マウント）になり、欠けた blob の補完 pull が
# `read-only file system` で即死する（v8 実測）。
import glob as _g, json as _json

def _stage(name, extra_roots=()):
    """DATASET_PATH/name 以下のファイルを実dirとして配置（file symlink）。

    patch Dataset の blobs（config blob sha256-cbe6ad4… 214B が入っている）
    も合流させる。これが無いと `ollama show` が失敗して毎回 pull 誘発。
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
            # manifests は「実ファイル」のみ正。symlink は v11 の残骸なので
            # 作り直す（skip すると同一セッションの再実行で修正が効かない）
            if name == "manifests":
                if os.path.lexists(d) and not os.path.islink(d) and os.path.isfile(d):
                    continue
                if os.path.lexists(d):
                    os.remove(d)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                # ollama は manifest 自体を symlink 検査し、対象が sha256 blob で
                # ないと拒否する（v11 実測 "manifest symlink target ... is not a
                # sha256 blob" → show 失敗 → pull 誘発）→ 1KB 級なので実体コピー
                shutil.copyfile(p, d)
                n += 1
                continue
            # blobs: 有効な symlink/実ファイルだけスキップ、壊れていれば作り直す
            # （lexists は壊れた symlink も True なので isfile で見分ける）
            if os.path.lexists(d) and os.path.isfile(d):
                continue
            if os.path.lexists(d):
                os.remove(d)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                os.symlink(p, d)
            except OSError:
                # 16 GiB を /kaggle/tmp へ複製すると disk full。巨大ファイルは
                # 複製せず落とす（symlink 不可 = 環境異常なので即中断が正解）
                if os.path.getsize(p) > (1 << 30):
                    raise
                shutil.copy2(p, d)
            n += 1
    return dst, n

if os.path.isdir(DATASET_PATH) and os.listdir(DATASET_PATH):
    print("📁 Dataset 検出 → モデルデータを準備中...")
    # 失敗時は例外のまま死なせず、理由を FAILED.txt に残す
    try:
        _, n_blob = _stage("blobs", extra_roots=(PATCH_PATH,))
        _, n_man = _stage("manifests")
    except OSError as _e:
        fail(f"モデルデータ配置に失敗: {type(_e).__name__}: {_e}")
    print(f"✅ blobs {n_blob} 件 / manifests {n_man} 件を配置（patch の config blob 含む）")

    # manifest が要求する blob の充足を先に点検。欠けたまま `ollama show` を
    # 叩くと pull が走り、v8 の失敗経路（16.5 GiB 再DL）へ戻る。
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
            print(f"⚠ 欠落 blob {len(_miss)} 件 → pull フォールバックの可能性: {_miss[:5]}")
        else:
            print("✅ manifest 要求 blob 全数充足 → pull 不要")
    else:
        print(f"⚠ manifest が見つからない: {_man}")
else:
    print(f"⚠ Dataset が見つかりません: {DATASET_PATH}")
    print("  → Add Data で Dataset を追加するか、下で直接 pull します。")

# --- 環境変数 ---
# OLLAMA_CONTEXT_LENGTH : コンテキスト長（未設定だと 4096 に落ちる）
# OLLAMA_KV_CACHE_TYPE  : KV を q8_0 に量子化して VRAM を半分に
# OLLAMA_FLASH_ATTENTION: 有効化
# OLLAMA_KEEP_ALIVE     : -1 で常時ロード維持
env = {
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

# --- 既存プロセスを掃除 ---
# 前回セッションの孤児 llama-server が残っているとポート/VRAMを
# 奪って本番と競合する。SIGTERM → 待機 → 残存分は SIGKILL。
subprocess.run(["pkill", "-TERM", "-f", "llama-server"], capture_output=True)
time.sleep(3)
subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
time.sleep(2)

OLLAMA_BIN = "/usr/local/bin/ollama"
with open("/tmp/ollama.log", "w") as log:
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)
print(f"🔄 Ollama PID: {proc.pid}")

# --- 起動待機 ---
for i in range(60):
    r = subprocess.run([OLLAMA_BIN, "list"], env=env, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ Ollama サーバー起動完了（{i+1}s）")
        break
    time.sleep(1)
else:
    print(open("/tmp/ollama.log").read()[-3000:])
    fail("Ollama サーバー起動失敗", "/tmp/ollama.log")

# --- モデル確認 ---
r = subprocess.run([OLLAMA_BIN, "show", MODEL_NAME], env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    # なぜ show が失敗したかを残さないと「全数充足なのに pull」が再現不能になる
    print(f"⚠ show 失敗（stderr）: {r.stderr.strip()[:500]}")
    print(f"⏳ Dataset にモデルが無いため pull します（16.5 GiB / 10〜30分）")
    # pull 前に空き容量を確認。16.5 GiB + 退避バッファが無いと
    # 途中で ENOSPC になり、長い待機の後に初めて失敗するのを防ぐ。
    _du = shutil.disk_usage("/kaggle/tmp")
    _need = 18 * (1 << 30)   # 16.5 GiB + 余裕
    print(f"   disk /kaggle/tmp: free={_du.free/(1<<30):.1f} GiB "
          f"(need ~{_need/(1<<30):.0f} GiB)")
    if _du.free < _need:
        fail(f"/kaggle/tmp 空き不足: {_du.free/(1<<30):.1f} GiB "
             f"< 必要 {_need/(1<<30):.0f} GiB")
    # 失敗しても例外のまま死なず、証跡を残してから止める
    # （進捗はセル出力へそのまま流す。pull のエラー表示もそこに残る）
    try:
        pull = subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=env,
                              timeout=7200)
    except subprocess.TimeoutExpired:
        fail("モデル pull が 7200s でタイムアウトしました")
    if pull.returncode != 0:
        fail("モデル pull に失敗しました（詳細はこのセルの出力）")
    print("✅ モデルダウンロード完了")
else:
    print(f"✅ モデル {MODEL_NAME} は利用可能")

print()
print(subprocess.run([OLLAMA_BIN, "list"], env=env,
                     capture_output=True, text=True).stdout)
'''))

# ── 5: warmup ───────────────────────────────────────────────
cells.append(("markdown", "---\n### 🔥 ウォームアップ（初回ロードは数分かかります）\n"))

cells.append(("code", '''# モデルを VRAM にロードして疎通確認
# 初回は 2〜5 分かかります（16.5 GiB をロード）
import json, time, urllib.request

def _post(path, payload, timeout=900):
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

print(f"⏳ ウォームアップ中（コンテキスト {CONTEXT_LENGTH:,}）...")
t0 = time.time()
try:
    resp = _post("/api/chat", {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": False,
        # 16 だと thinking（推論トークン）だけで予算を使い切り、
        # content が空のまま返る（実測 reply=''）。生成証明には十分な余量を。
        "options": {"num_predict": 128, "temperature": 0},
    })
    dt = time.time() - t0
    print(f"✅ 応答あり（{dt:.1f}s）")
    print(f"   load : {(resp.get('load_duration') or 0)/1e9:.1f}s")
    print(f"   reply: {resp.get('message', {}).get('content', '')!r}")
    ec = resp.get("eval_count") or 0
    ed = (resp.get("eval_duration") or 1) / 1e9
    print(f"   速度 : {ec/ed:.1f} tokens/s" if ec else "   速度 : N/A")
    if not (resp.get("message", {}).get("content")
            or resp.get("message", {}).get("thinking")):
        # HTTP 成功だが生成物が空 = verify [2] と同じ症状の予兆なので可視化する
        print("   ⚠ content/thinking が空（verify [2] で判定されます）")
    # GPU 常駐検査。CPU フォールバックでも HTTP 200 は返るので、
    # 通信だけ見ていると 3/3 PASS のまま静かに劣化する。
    import subprocess as _sp
    _ps = _sp.run([OLLAMA_BIN, "ps"], env=env,
                  capture_output=True, text=True).stdout.strip().splitlines()
    print("   ps :", " | ".join(l.strip() for l in _ps[1:] if l.strip()) or "(空)")
    _psrow = " ".join(l for l in _ps[1:] if "%" in l)
    if _psrow and "GPU" not in _psrow:
        print("   ⚠ プロセッサ表示に GPU が無い → CPU 実行の可能性")
    # ollama ps の CONTEXT 列はカンマ無し（131072）。正規化して比較する。
    if _psrow and str(CONTEXT_LENGTH) not in _psrow.replace(",", ""):
        print(f"   ⚠ コンテキストが {CONTEXT_LENGTH:,} と異なる（ps 表示: {_psrow}）")
except Exception as e:
    print(f"⚠ ウォームアップ失敗: {e}（LOAD_TIMEOUT={env.get('OLLAMA_LOAD_TIMEOUT')}s）")
    print(open("/tmp/ollama.log").read()[-3000:])
    # 握り潰すと verify が「原因不明の FAIL」になるだけなので、ここで止める
    fail(f"ウォームアップ失敗: {type(e).__name__}: {e}", "/tmp/ollama.log")
'''))

# ── 6: トンネル（cloudflared / ngrok）─────────────────────────
# セル構成は共通にし、トンネルセルだけ差し替える。共通インターフェース:
#   TUNNEL_NAME   : pgrep 用のプロセス名
#   TUNNEL_LOG    : ログ出力先
#   RESTART_CMD   : 再起動用の argv（keepalive が使う）
#   probe_url()   : 現在の公開URLを返す（無ければ ""）
if IS_NGROK:
    cells.append(("markdown",
                  "---\n### 🔗 ngrok トンネル → 公開URL\n\n"
                  "Secrets の `NGROK_TOKEN` を使います。`NGROK_DOMAIN` を設定すると\n"
                  "**毎回同じ URL** になります（未設定ならランダム）。\n"))
    cells.append(("code", '''# ngrok 起動 → 公開URL取得（authtoken は Secrets の NGROK_TOKEN）
import json, os, subprocess, time, urllib.request

NGROK_BIN = "/usr/local/bin/ngrok"
NGROK_API = "http://127.0.0.1:4040/api/tunnels"

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
# 既定値を埋めているので、未設定でも必ず固定URLになる（Secret は上書き用）。
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
# --pooling-enabled が無いと、同じ固定ドメインが前回のセッションから
# 残っているときに ERR_NGROK_334 で即死する（実測）。
# プーリングを有効にすると複数エンドポイントを束ねられるので衝突しない。
RESTART_CMD = [NGROK_BIN, "http", str(PROXY_PORT),
               "--log=stdout", "--pooling-enabled"]
if NGROK_DOMAIN:
    # --domain は deprecated（実測で警告）。--url が現行。
    RESTART_CMD.append("--url=https://" + NGROK_DOMAIN)


def probe_url():
    """ngrok のローカル管理APIから公開URLを取る（ログ解析より確実）。"""
    try:
        with urllib.request.urlopen(NGROK_API, timeout=3) as r:
            d = json.loads(r.read().decode())
        for t in d.get("tunnels", []):
            u = t.get("public_url", "")
            if u.startswith("https://"):
                return u
    except Exception:
        pass
    return ""


# 前回のトンネルを確実に落とす（多重起動すると URL が混ざる）
subprocess.run(["pkill", "-x", "ngrok"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as _log:
    ng_proc = subprocess.Popen(RESTART_CMD, stdout=_log, stderr=_log)
print("🔄 ngrok PID: " + str(ng_proc.pid))

public_url = ""
for i in range(60):
    public_url = probe_url()
    if public_url:
        break
    if i % 10 == 9:
        print("⏳ URL 待ち (" + str(i + 1) + "/60)...")
    time.sleep(1)

if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("ngrok の公開URL取得に失敗しました", TUNNEL_LOG)

print("🌍 URL: " + public_url)
print("⏳ トンネル確立待ち（最大 60s）...")
import urllib.error
ready = False
for i in range(20):
    try:
        with urllib.request.urlopen(public_url + "/healthz", timeout=10) as r:
            if r.status == 200:
                print("✅ トンネル確立（" + str((i + 1) * 3) + "s）")
                ready = True
                break
    except Exception:
        pass
    time.sleep(3)

if not ready:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\\n")
print()
print("=" * 60)
print("  🌍 PUBLIC URL : " + public_url)
print("=" * 60)
'''))
else:
    cells.append(("markdown", "---\n### 🔗 cloudflared トンネル → 公開URL\n"))
    cells.append(("code", '''# cloudflared quick tunnel 起動 → 公開URL取得
# アカウント不要。authtoken も Secrets も要らない。
import os, re, subprocess, time

CF_BIN = "/usr/local/bin/cloudflared"
TUNNEL_NAME = "cloudflared"
TUNNEL_LOG = "/tmp/cloudflared.log"
RESTART_CMD = [CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]
URL_RE = re.compile(r"https://[a-z0-9-]+\\.trycloudflare\\.com")


def probe_url():
    """cloudflared のログから公開URLを取る。"""
    try:
        with open(TUNNEL_LOG) as f:
            m = URL_RE.search(f.read())
        return m.group(0) if m else ""
    except FileNotFoundError:
        return ""


# 前回のトンネルを確実に落とす（多重起動すると URL が混ざる）
subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)

with open(TUNNEL_LOG, "w") as log:
    cf_proc = subprocess.Popen(RESTART_CMD, stdout=log, stderr=log)
print(f"🔄 cloudflared PID: {cf_proc.pid}")

# ---- 公開URL取得（ログに URL が出るまで待つ）----
public_url = ""
for i in range(90):
    public_url = probe_url()
    if public_url:
        break
    if i % 10 == 9:
        print(f"⏳ URL 待ち ({i+1}/90)...")
    time.sleep(1)

if not public_url:
    # ここで止めないと後段が /tmp/public_url.txt なしでエラーになり、
    # どこで失敗したか分からなくなる。原因ログも永続領域に残す。
    print("⚠ 公開URLを取得できませんでした")
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("cloudflared の公開URL取得に失敗しました", TUNNEL_LOG)

# ---- トンネル確立まで待つ ----
# URL が出た直後はまだエッジに伝播しておらず、外から叩くと 000 になる。
# ここで待たないと次のセルの疎通確認が誤って失敗する。
print(f"🌍 URL: {public_url}")
print("⏳ トンネル確立待ち（最大 60s）...")
import urllib.request, urllib.error
ready = False
for i in range(20):
    try:
        with urllib.request.urlopen(
            f"{public_url}/healthz", timeout=10
        ) as r:
            if r.status == 200:
                print(f"✅ トンネル確立（{(i+1)*3}s）")
                ready = True
                break
    except Exception:
        pass
    time.sleep(3)

if not ready:
    print("⚠ 確立を確認できませんでした。verify セルで再確認します。")

with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
# 完了後に `kaggle kernels output` で取り出す用
save_output("public_url.txt", public_url + "\\n")
print()
print("=" * 60)
print(f"  🌍 PUBLIC URL : {public_url}")
print("=" * 60)
'''))


# ── 6.5: 固定URL配布（Gist）─────────────────────────────────
cells.append(("markdown", "---\n### 📌 固定URL配布（quick tunnel の URL を Gist に反映）\n\n"
              "quick tunnel の URL は毎回変わるので、**取得先だけ固定**します。\n"
              "クライアントは `https://gist.githubusercontent.com/.../raw/url.txt` を\n"
              "1回 GET して現在の URL を得ます（GITHUB_TOKEN 未設定ならスキップ）。\n"))

cells.append(("code", '''# 現在の公開URLを Gist に書き出す（取得先の固定）
# Secrets: GITHUB_TOKEN（gist 権限のみでよい）, GIST_ID（省略時は既定 gist）
# keepalive で cloudflared が再起動したときも同じ関数を呼ぶ。
import json, os, urllib.request

GIST_ID_DEFAULT = "2b44ac8cb115012a31d3c46661af933f"
GIST_OWNER = "yamahei21python"
# 環境変数 → Secrets → 既定値（GITHUB_TOKEN と同じ解決順に揃える）
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
    """現在の公開URL を Gist に書く（cloudflared は URL が毎回変わるため）。

    Gist は secret（非公開URL）だが、URL を知れば誰でも読める。
    認証は付けていないので、実質この URL 自体が入口になる。
    """
    tok = _gh_token()
    if not tok:
        print("ℹ GITHUB_TOKEN 未設定 → 固定URL配布はスキップ（接続情報セルを使う）")
        return False
    files = {"url.txt": {"content": u + "\\n"}}
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
      + GIST_OWNER + "/" + GIST_ID + "/raw/url.txt")
'''))

# ── 7: verify ───────────────────────────────────────────────
cells.append(("markdown", "---\n### ✅ 公開URL経由の疎通確認\n"))

cells.append(("code", '''# 公開URL経由の疎通確認（UI / 200 / streaming / OpenAI互換 / GPU常駐の5点）
# 結果は /kaggle/working/verify_result.txt に残す（save_output が /output にも複製）。
# 失敗は raise（検証不能のまま放置しない）。
import json, time as _time, urllib.request, urllib.error

try:
    with open("/tmp/public_url.txt") as f:
        public_url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

# ---- トンネル確立の取り直し ----
# tunnel セルでエッジ伝播待ちが終わっていなくても、ここで吸収する。
# 届かなければ [1] がそのまま FAIL になる（握り潰さない）。
for i in range(10):
    try:
        with urllib.request.urlopen(f"{public_url}/healthz", timeout=10) as r:
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
    f"MODEL={MODEL_NAME}",
    f"TIME={_time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]
failures = []


def _call(url, timeout=900, stream=False):
    # ngrok 無料枠のブラウザ向け警告ページを踏まないためのヘッダ（cloudflared では無害）
    headers = {"Content-Type": "application/json",
               "ngrok-skip-browser-warning": "true"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": stream,
        # 非ストリームの 16 は thinking だけで枯渇し content が空になる
        # （実測 reply=''）。十分な余量を与える。
        "options": {"num_predict": 128 if not stream else 64, "temperature": 0},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _judge(idx, label, ok, detail):
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _with_retry(fn, tries=6, delay=5):
    # 実行時ログで確認した瞬間的障害: URLError [Errno 101] Network is
    # unreachable / [-2] Name or service not known（v11 で verify [2][3] が
    # 約1分の DNS/経路瞬断で落ち、[1][4] は通過＝経路は生きていた）。
    # 3回/4秒（~8秒）では吸収できないので、待ちを 5,10,15,20,25 秒と
    # 伸ばして合計 75 秒の窓を持つ。HTTP 応答（401/500 等）は「答え」
    # なので再試行せず、そのまま判定に渡す。
    last = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            last = e
            if i + 1 < tries:
                wait = delay * (i + 1)
                print(f"  ⚠ 一時的な通信エラー（{i+1}/{tries}）: {e} "
                      f"→ {wait}s 待って再試行")
                _time.sleep(wait)
    raise last


# [1] 認証なしでチャット UI が取れる（/ は 200 で HTML）
print("[1] 認証なしで UI")
try:
    with _with_retry(lambda: urllib.request.urlopen(
            f"{public_url}/", timeout=60)) as r:
        html = r.read()
        # UI 判定は HTML であること + 十分なサイズで行う。
        # 旧実装は b"Kaggle Chat" を見ていたが、HTML 側のタイトル変更で
        # 常に FAIL になる（実測）。文言に依存させない。
        ok1 = (r.status == 200 and len(html) > 10000
               and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
        _judge(1, "ui", ok1, f"HTTP {r.status} {len(html)}B")
except Exception as e:
    _judge(1, "ui", False, f"{type(e).__name__}: {e}")

# [2] 認証なしで /api/chat が 200 + 生成
print("[2] /api/chat")
try:
    with _with_retry(lambda: _call(f"{public_url}/api/chat")) as r:
        resp = json.loads(r.read())
        reply = (resp.get("message") or {}).get("content", "")
        think = (resp.get("message") or {}).get("thinking", "")
        # 生成証明 = content か thinking か評価トークンのどれかが実在する
        #（thinking モデルは content が空でも生成は成功している）。
        made = bool(reply or think or (resp.get("eval_count") or 0))
        _judge(2, "chat-ok", r.status == 200 and made,
               f"HTTP {r.status} reply={reply!r} "
               f"eval={resp.get('eval_count')}")
except Exception as e:
    _judge(2, "auth-ok", False, f"{type(e).__name__}: {e}")

# [3] ストリーミング（chunked が cloudflared を通り抜けるか）
#     ここが通らなければ token 逐次出力 = 通常の利用形態が壊れる。
print("[3] ストリーミング")
try:
    times = []
    # 接続確立を先に終えてから計測を始める（再試行の sleep が first/total に
    # 混ざると逐次判定が偽 FAIL になる）
    resp3 = _with_retry(lambda: _call(f"{public_url}/api/chat",
                                      timeout=600, stream=True))
    t0 = _time.time()
    with resp3 as r:
        for raw in r:            # 改行が届くまで待つ = 逐次受信になっている証拠
            if not raw.strip():
                continue
            times.append(_time.time() - t0)
    total = _time.time() - t0
    if len(times) >= 3:
        first = times[0]
        spread = times[-1] - first
        # 到着分布で判定（生成長に依存しない）:
        #  - 全バッファ転送なら first ≈ total（first < total*0.8 が偽）。
        #  - 3 到着以上 & 最初と最後の間隔が 0.2s 超 = 逐次の到着証拠。
        # span>0.5 の旧基準は短い生成で偽 FAIL になった。
        ok = spread > 0.2 and first < total * 0.8
        detail = (f"chunks={len(times)} first={first:.2f}s "
                  f"total={total:.2f}s spread={spread:.2f}s")
    else:
        ok, detail = False, f"chunks={len(times)} first=N/A"
    _judge(3, "streaming", ok, detail)
except Exception as e:
    _judge(3, "streaming", False, f"{type(e).__name__}: {e}")

# [4] OpenAI 互換経路（配布する opencode.json が使うのはこちら）
print("[4] /v1/chat/completions")
try:
    req4 = urllib.request.Request(
        f"{public_url}/v1/chat/completions",
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 128,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "ngrok-skip-browser-warning": "true"},
        method="POST",
    )
    with _with_retry(lambda: urllib.request.urlopen(req4, timeout=300)) as r:
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

# [5] GPU 常駐 + コンテキスト長（CPU フォールバックの検出）
#     通信だけ見ていると 3/3 PASS のまま VRAM に載っていないことがある。
print("[5] GPU 常駐")
try:
    import subprocess as _sp
    ps_out = _sp.run(["/usr/local/bin/ollama", "ps"],
                     env=env, capture_output=True, text=True, timeout=30).stdout
    # データ行だけに絞る（ヘッダは "GPU" 文字を含まない）。
    # ollama ps の CONTEXT 列はカンマ無し（例: 131072）なので両方正規化して比較。
    rows = [l.strip() for l in ps_out.splitlines()[1:] if "%" in l]
    row = " | ".join(rows)
    # 完全 CPU フォールバックは "100% CPU" で GPU 文字が無い。
    # 分割オフロード（x% GPU/y% CPU）は VRAM 常駐の一部として許容する。
    has_gpu = "GPU" in row
    has_ctx = str(CONTEXT_LENGTH) in row.replace(",", "")
    _judge(5, "gpu-resident", bool(row) and has_gpu and has_ctx,
           f"ps={row[:160]!r} ctx_ok={has_ctx}")
except Exception as e:
    _judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

# ---- レポート ----
report = "\\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\\n"):
    # 保存失敗 = 証跡なし。ログは取れないので成功扱いにしてはいけない。
    failures.append("verify_result.txt を永続領域に保存できなかった")

if failures:
    fail("疎通確認に失敗: " + "; ".join(failures), TUNNEL_LOG)
print("✅ 5/5 パス — UI / 200 / streaming / OpenAI互換 / GPU常駐")
'''))

# ── 8: connection info ──────────────────────────────────────
cells.append(("markdown", "---\n### 📋 接続情報\n"))

cells.append(("code", '''# 接続情報（クライアント側の設定にコピペしてください）
# セル出力・ログは後から取れないので /kaggle/working に保存する（/output にも複製）。
import json

try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

opencode_cfg = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "kaggle": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Kaggle Ollama",
            "options": {"baseURL": f"{url}/v1"},
            "models": {MODEL_NAME: {"name": "Qwen3.8-27B-Uncensored"}},
        }
    },
    "model": f"kaggle/{MODEL_NAME}",
}

buf = [
    "=" * 64,
    f"  🌍 Public URL : {url}",
    f"  💬 チャット UI : {url}/",
    f"  🤖 Model      : {MODEL_NAME}",
    f"  📏 Context    : {CONTEXT_LENGTH:,}",
    "  🔓 認証        : なし（URL を知れば誰でも使える）",
    "=" * 64,
    "",
    "📋 curl:",
    f"  curl {url}/api/tags",
    "",
    "📋 Python（OpenAI 互換クライアント）:",
    "  import httpx",
    "  from openai import OpenAI",
    "  client = OpenAI(",
    f"      base_url='{url}/v1',",
    "      api_key='dummy',",
    "  )",
    f"  client.chat.completions.create(model='{MODEL_NAME}', messages=[...])",
    "",
    "📋 opencode.json:",
    json.dumps(opencode_cfg, indent=2, ensure_ascii=False),
    "",
    "⚠ cloudflared quick tunnel: 稼働保証なし。切れたらトンネルセルを再実行",
    "⚠ 公開URLは実行のたびに変わる。再実行後は接続情報を取り直す",
    "⚠ Kaggle: 9時間でカーネルが停止します",
]

report = "\\n".join(buf)
print(report)
if not save_output("connection_info.txt", report + "\\n"):
    # 接続情報は唯一の利用手掛かりなので、取れなければ成功にしない
    fail("connection_info.txt を永続領域に保存できませんでした")
'''))

# ── 9: keepalive ────────────────────────────────────────────
cells.append(("markdown", """---
### 🔋 キープアライブ

既定 **5 分**（検証用）で自動終了します（`KEEPALIVE_MINUTES` で変更、上限 480）。

> **なぜ無限ループにしないのか**
> 実行が完了しないと `/kaggle/working` の zip が生成されず、**ログも成果物も一切取得できません**。
> 検証は必ず完了させてから `kaggle kernels output` で行ってください。

```bash
kaggle kernels output yamahei21python/step2-orcarouter-qwen38-27b-cloudflared -p out/
cat out/verify_result.txt out/connection_info.txt
```
"""))

cells.append(("code", '''# キープアライブ: 有限ループ（完了させると /kaggle/working の zip を取得できる）
# 無限ループだと実行が完了せず、ログも output も構造的に取れないため有限にしてある。
# 検証中は 5 分、完成後の運用は 480 分（Kaggle の 9h 上限より短い最大値）。
import os, subprocess, time, re, json, urllib.request, urllib.error

KEEPALIVE_MINUTES = 5         # 検証用。完成したら 480 に戻す
MAX_MINUTES = 480             # これを超えると成果物ごと消えるので実装でも守る
TUNNEL_CHECK_EVERY = 10       # 公開URL疎通は 10 分ごと（外部リクエスト削減）

# 0以下は「維持しない」意図だが、丸めて 480 にすると意図せず8時間固定になる。
# 誤設定を黙って膨らませないため、0以下は即 fail（証跡を残して止める）。
if KEEPALIVE_MINUTES <= 0:
    fail(f"KEEPALIVE_MINUTES={KEEPALIVE_MINUTES} は無効（1〜{MAX_MINUTES} で指定）")
# 記述だけ上限を書くと 600 などに書き換えた人が成果物を失うので、ここで丸める
if KEEPALIVE_MINUTES > MAX_MINUTES:
    print(f"⚠ KEEPALIVE_MINUTES={KEEPALIVE_MINUTES} → {MAX_MINUTES} に制限")
    KEEPALIVE_MINUTES = MAX_MINUTES

KEEPALIVE_CONF = "/tmp/keepalive.conf"   # ON/OFF + MINUTES の共有ファイル
KEEPALIVE_KEEPALIVE = 60                 # チャット送信ごとにリセットする時間（分）


def _read_keepalive_conf():
    """RUN=1/0 と MINUTES を読み取り（デフォルトは 1 と KEEPALIVE_KEEPALIVE）。"""
    try:
        with open(KEEPALIVE_CONF) as f:
            lines = [l.strip() for l in f if l.strip() and "=" in l]
        d = dict(l.split("=", 1) for l in lines)
        run = int(d.get("RUN", 1))
        minutes = int(d.get("MINUTES", KEEPALIVE_KEEPALIVE))
    except FileNotFoundError:
        run, minutes = 1, KEEPALIVE_KEEPALIVE
    except Exception:
        run, minutes = 1, KEEPALIVE_KEEPALIVE
    if run <= 0:
        return 0, KEEPALIVE_KEEPALIVE   # RUN<=0 は「維持しない」
    if minutes <= 0:
        fail(f"KEEPALIVE_MINUTES={minutes} は無効（1〜{MAX_MINUTES} で指定）")
    if minutes > MAX_MINUTES:
        print(f"⚠ KEEPALIVE_MINUTES={minutes} → {MAX_MINUTES} に制限")
        minutes = MAX_MINUTES
    return run, minutes


def _reset_keepalive():
    """チャット送信ごとに時間リセット（ON 維持）。"""
    try:
        with open(KEEPALIVE_CONF, "w") as f:
            f.write(f"RUN=1\\nMINUTES={KEEPALIVE_KEEPALIVE}\\n")
    except Exception:
        pass


try:
    with open("/tmp/public_url.txt") as f:
        url = f.read().strip()
except FileNotFoundError:
    url = ""


def _proc(name):
    """プロセス生死。pid があるだけで生存とは限らないので tunnel 疎通と併用。"""
    try:
        # ollama は -f だと子の `llama-server -m ...ollama_models...` まで
        # 全一致して偽 alive になる。プロセス名完全一致で見る。
        args = ["pgrep", "-x", name] if name == "ollama" else ["pgrep", "-f", name]
        out = subprocess.run(args, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return "alive" if out else "DEAD"
    except Exception:
        return "unknown"


def _gpu():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().replace("\\n", " | ")
    except Exception:
        return "N/A"


print(f"Public URL: {url}")
RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()
if RUN:
    print(f"🔋 ON。{KEEPALIVE_KEEPALIVE}分間維持（チャット送信でリセット、■ で停止）")
else:
    print(f"🛑 ON/OFF OFF（ RUN=0 ）。即時終了します")
    KEEPALIVE_MINUTES = 0
print("完了後: kaggle kernels output <slug> -p <dir> で成果物を取得")
print()

# ログ API は取れないため、毎分の行も**ここに溜めて** zip に残す
def _conf_mtime():
    """keepalive.conf の更新時刻。proxy がチャット送信ごとに書き直す。"""
    try:
        return os.path.getmtime(KEEPALIVE_CONF)
    except OSError:
        return 0.0


log = []
ollama_st = cf_st = "unknown"
cf_restarted = 0
tunnel_dead = 0
_restarted_now = False
# 1周も回らなかった場合（RUN=0 即時終了）は死活判定をしない。
# ループ未実行だと ollama_st/cf_st が "unknown" のままになり、
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
    RUN, KEEPALIVE_MINUTES = _read_keepalive_conf()   # 都度 ON/OFF + 時間確認
    if not RUN:
        print(f"RUN=0 で {i}周目にキープアライブ終了")
        break
    _m = _conf_mtime()
    if _m != conf_mtime:
        conf_mtime = _m
        limit = i - 1 + KEEPALIVE_MINUTES
        print(f"♻ アクティビティ検出 → 残り {KEEPALIVE_MINUTES} 分に延長")
    ran_any = True
    time.sleep(60)
    ollama_st, cf_st = _proc("ollama"), _proc(TUNNEL_NAME)

    # トンネルが死んだら1回だけ起こす（cloudflared は URL が変わるので
    # 新 URL を取り直して public_url.txt を書き換える。2回目以降は fail 対象）
    # 再起動コマンドと URL 取得はトンネルセルが定義した共通インターフェースを使う。
    if cf_st == "DEAD" and cf_restarted == 0:
        cf_restarted = 1
        try:
            subprocess.run(["pkill", "-x", TUNNEL_NAME], capture_output=True)
            time.sleep(1)
            with open(TUNNEL_LOG, "w") as _lg:
                subprocess.Popen(RESTART_CMD, stdout=_lg, stderr=_lg)
            _new_url = ""
            for _ in range(30):          # 最大30秒で新URL待ち
                time.sleep(1)
                _new_url = probe_url()
                if _new_url:
                    break
            if _new_url:
                url = _new_url
                with open("/tmp/public_url.txt", "w") as _f:
                    _f.write(url + "\\n")
                # zip 側も更新しないと、成果物には失効URLが残る
                save_output("public_url.txt", url + "\\n")
                print(f"♻ {TUNNEL_NAME} 再起動 → 新URL: {url}")
                # 固定URL配布（Gist）も更新しないと、クライアントは失効URLを掴む
                if "publish_url" in globals():
                    publish_url(url)
                cf_st = "alive(restarted)"
                # 新トンネルは別勘定。エッジ伝播待ち中の /healthz 失敗を
                # 旧カウントに混ぜると、稼働中でも次回チェックで偽FAILになる。
                tunnel_dead = 0
                _restarted_now = True
        except Exception as _e:
            print(f"⚠ cloudflared 再起動失敗: {type(_e).__name__}: {_e}")

    line = (f"[{time.strftime('%H:%M:%S')}] "
            f"ollama={ollama_st} {TUNNEL_NAME}={cf_st} | GPU: {_gpu()}")

    # pid だけでは tunnel の死は分からない（quick tunnel は URL 失効がありうる）
    # 再起動した同一イテレーションは新URLのエッジ伝播待ちなので判定しない
    if url and i % TUNNEL_CHECK_EVERY == 0 and not _restarted_now:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=15) as r:
                line += " | tunnel=OK" if r.status == 200 else f" | tunnel=HTTP{r.status}"
        except Exception as e:
            tunnel_dead += 1
            line += (f" | tunnel=DEAD({type(e).__name__}) x{tunnel_dead}")
            if cf_st == "alive":
                line += " ← トンネル生存中だが URL 失効の可能性"
    _restarted_now = False

    log.append(line)
    print(line)

# どれか1つでも死んでいれば失敗扱い（tunnel は pid と併せて判定）。
# tunnel は1回の瞬断で全体を FAIL にしない（30分稼働後に成果物が死なないように）。
# 2回以上 = 恒久的な失効とみなし、cf再起動が効かなかった場合もこれで拾う。
# unknown（pgrep 例外/timeout）は確認不能 = 安全側で失敗扱いにする。
# ただし 1 周も回っていない（RUN=0 即時終了）場合は判定材料が無いので fail しない。
failed = False
if not ran_any:
    ollama_st, cf_st = _proc("ollama"), _proc(TUNNEL_NAME)
    print(f"ℹ 即時終了のため死活判定はスキップ（ollama={ollama_st} "
          f"{TUNNEL_NAME}={cf_st}）")
else:
    failed = (ollama_st in ("DEAD", "unknown") or cf_st in ("DEAD", "unknown")
              or tunnel_dead >= 2)
report = "\\n".join([
    f"result={'DEAD' if failed else ('stopped' if not RUN else 'alive')}",
    f"RUN={RUN}",
    f"minutes={KEEPALIVE_MINUTES}",
    f"ran_any={ran_any}",
    f"ollama={ollama_st}",
    f"tunnel={TUNNEL_NAME}:{cf_st}",
    f"url={url}",
    "",
] + log)
# 保存に失敗すると「生きている証跡」が残らないので、失敗も 1 件は証跡を残す
saved = save_output("keepalive.txt", report + "\\n")
print()
print("⏱ キープアライブ終了。COMPLETE で /kaggle/working が zip 化されます")

# ============================================================
# 後片付け: GPU を明示的に手放す
# ============================================================
# これをやらないと、OLLAMA_KEEP_ALIVE=-1 で 23 GiB が VRAM に
# 固定されたまま残り、カーネルが COMPLETE になっても GPU を
# 掴み続ける（実測: expires_at が 2319 年になる）。
# プロセスを落としても Kaggle 側の枠開放は保証されないが、
# 少なくとも VRAM は即座に返る。
print()
print("=" * 56)
print("🧹 後片付け: GPU / プロセスを解放")
print("=" * 56)


def _sh(cmd, label):
    """シェルコマンドを実行して成否を 1 行で出す。"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0
        print(f"  {'✅' if ok else '— '} {label}")
        return ok
    except Exception as e:
        print(f"  ⚠ {label}: {type(e).__name__}")
        return False


def _pids(pattern):
    """pgrep で該当 PID を返す（無ければ空リスト）。"""
    try:
        r = subprocess.run(f"pgrep {pattern}", shell=True,
                           capture_output=True, text=True, timeout=15)
        return [p for p in r.stdout.split() if p.strip()]
    except Exception:
        return []


# --- 1) モデルを明示的にアンロード（keep_alive=0）---
# これが一番効く。プロセスを殺すより先に、Ollama 自身に VRAM を返させる。
unloaded = False
for path, payload in [
    ("/api/generate", {"model": MODEL_NAME, "keep_alive": 0}),
    ("/api/chat", {"model": MODEL_NAME, "messages": [], "keep_alive": 0}),
]:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status == 200:
                unloaded = True
                print(f"  ✅ アンロード要求 OK ({path})")
                break
    except Exception as e:
        print(f"  —  {path}: {type(e).__name__}")

# --- 2) VRAM が実際に空いたか確認 ---
for attempt in range(10):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/ps", timeout=10) as r:
            models = json.loads(r.read())
        loaded = models.get("models") or []
        if not loaded:
            print("  ✅ VRAM 解放確認（常駐モデル 0）")
            break
        vram = sum(m.get("size_vram", 0) for m in loaded)
        print(f"  ⏳ まだ常駐: {vram:,} B（{attempt + 1}/10）")
    except Exception as e:
        print(f"  —  /api/ps 確認失敗: {type(e).__name__}")
        break
    time.sleep(3)
else:
    print("  ⚠ アンロードを確認できず（プロセス kill へ進む）")

# --- 3) プロセスを落とす（SIGTERM → 待機 → SIGKILL）---
# 順序: 外側（トンネル）→ 内側（プロキシ）→ 本体（ollama）
TARGETS = [
    ("-x ngrok", "ngrok"),
    ("-x cloudflared", "cloudflared"),
    ("-f auth_proxy.py", "auth_proxy"),
    ("-x ollama", "ollama"),
    ("-f llama-server", "llama-server"),
]

for pattern, label in TARGETS:
    if _pids(pattern):
        subprocess.run(f"pkill -TERM {pattern}", shell=True, capture_output=True)
        print(f"  TERM → {label}")
time.sleep(3)

# 残存分は SIGKILL（llama-server は ollama の子で別名なので個別に落とす）
for pattern, label in TARGETS:
    if _pids(pattern):
        subprocess.run(f"pkill -KILL {pattern}", shell=True, capture_output=True)
        print(f"  KILL → {label}（残存していた）")

# --- 4) 最終確認 ---
print()
time.sleep(2)
left = [label for pattern, label in TARGETS if _pids(pattern)]

if left:
    print(f"  ⚠ 残存: {', '.join(left)}")
else:
    print("  ✅ 全プロセス停止（ollama / tunnel / proxy / llama-server）")

# 後片付けの結果も証跡として残す
save_output("cleanup.txt", "\\n".join([
    f"unload_requested={unloaded}",
    f"leftover={','.join(left) if left else 'none'}",
    "",
]) + "\\n")
print("  📄 cleanup.txt 保存済")
print("=" * 56)

if failed:
    fail("死活確認に失敗しました（keepalive.txt を確認）", "/tmp/ollama.log")
elif not saved:
    fail("keepalive.txt を保存できませんでした（成果物ディレクトリを確認）")
'''))

# ── 書き出し ────────────────────────────────────────────────
nb = {
    "cells": [
        {
            "cell_type": t,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"execution_count": None, "outputs": []} if t == "code" else {}),
        }
        for t, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}: {len(cells)} cells")

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
with open(meta_path, "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"wrote {meta_path}")
