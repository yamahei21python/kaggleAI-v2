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
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import kcconfig  # noqa: E402

CFG = kcconfig.load()

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

# 定数だけのコードセルは kaggle/step2/cells/*.py に実ファイルとして置く
# （ruff などで構文チェックできるように）。f-string で値を埋め込むセルはここに残す。
CELLS_DIR = HERE / "step2" / "cells"


def _cell(name):
    return (CELLS_DIR / name).read_text(encoding="utf-8")


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

cells.append(("code", _cell("01_restore_binaries.py")))

# ── 3: auth proxy ───────────────────────────────────────────
cells.append(("markdown", "---\n### 🌐 リバースプロキシ起動（UI 配信 + 同一オリジン化）\n"))

cells.append(("code", _cell("02_proxy.py")))

# ── 4: ollama serve ─────────────────────────────────────────
cells.append(("markdown", "---\n### 🔥 Ollama サーバー起動\n"))

cells.append(("code", _cell("03_ollama_serve.py")))

# ── 5: warmup ───────────────────────────────────────────────
cells.append(("markdown", "---\n### 🔥 ウォームアップ（初回ロードは数分かかります）\n"))

cells.append(("code", _cell("04_warmup.py")))

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
NGROK_DOMAIN_DEFAULT = "''' + CFG["NGROK_DOMAIN"] + '''"
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
    cells.append(("code", _cell("05_cloudflared.py")))


# ── 6.5: 固定URL配布（Gist）─────────────────────────────────
cells.append(("markdown", "---\n### 📌 固定URL配布（quick tunnel の URL を Gist に反映）\n\n"
              "quick tunnel の URL は毎回変わるので、**取得先だけ固定**します。\n"
              "クライアントは `https://gist.githubusercontent.com/.../raw/url.txt` を\n"
              "1回 GET して現在の URL を得ます（GITHUB_TOKEN 未設定ならスキップ）。\n"))

cells.append(("code", '''# 現在の公開URLを Gist に書き出す（取得先の固定）
# Secrets: GITHUB_TOKEN（gist 権限のみでよい）, GIST_ID（省略時は既定 gist）
# keepalive で cloudflared が再起動したときも同じ関数を呼ぶ。
import json, os, urllib.request

GIST_ID_DEFAULT = "''' + CFG["GIST_ID"] + '''"
GIST_OWNER = "''' + CFG["GIST_OWNER"] + '''"
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

cells.append(("code", _cell("06_verify.py")))

# ── 8: connection info ──────────────────────────────────────
cells.append(("markdown", "---\n### 📋 接続情報\n"))

cells.append(("code", _cell("07_connection_info.py")))

# ── 9: keepalive ────────────────────────────────────────────
cells.append(("markdown", """---
### 🔋 キープアライブ

最後のチャット送信から **60 分**（`KEEPALIVE_RESET_MINUTES`）で自動終了します。
送信のたびに延長されますが、セル開始からの合計は **480 分**（`MAX_MINUTES`）で打ち切ります。
UI の ■ ボタン（`/stop`）で即終了します。

> **なぜ無限ループにしないのか**
> 実行が完了しないと `/kaggle/working` の zip が生成されず、**ログも成果物も一切取得できません**。
> 検証は必ず完了させてから `kaggle kernels output` で行ってください。

```bash
kaggle kernels output yamahei21python/step2-orcarouter-qwen38-27b-cloudflared -p out/
cat out/verify_result.txt out/connection_info.txt
```
"""))

cells.append(("code", _cell("08_keepalive.py")))

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
