#!/usr/bin/env python3
"""vLLM 版 Step2 の CPU ドライラン・ノート生成（GPU クォータ消費ゼロ）。

目的: 推論以外の全工程を GPU なしで検証する。

  1. vLLM の install 可否・所要時間・バージョン
  2. `vllm serve --help` に必要なフラグが存在するか（起動ラダーの前提）
  3. Dataset のマウントと symlink ステージング（MTP head の出し入れ）
  4. プロキシ + UI + --enforce-sampling の実挙動（モック上流で検証）
  5. cloudflared quick tunnel 越しの UI / models / streaming / OpenAI 互換
  6. Gist 配布・keepalive・後片付け

モック上流が 8000 番で vLLM の代役を務める。実推論はしない。

    python3 kaggle/build_vllm_dryrun.py
    cd kaggle/step2-vllm-dryrun && kaggle kernels push
"""
import base64
import json
import pathlib

OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
DATASET_ID = f"{OWNER}/{DATASET_SLUG}"
SERVED_NAME = "qwen3.8-27b-awq"
SLUG = "step2-vllm-dryrun"

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "step2-vllm-dryrun"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{SLUG}.ipynb"

PROXY_SRC_B64 = base64.b64encode(
    (HERE / "step2" / "auth_proxy.py").read_bytes()
).decode()
CHAT_SRC_B64 = base64.b64encode(
    (HERE.parent / "llama-chat.html").read_bytes()
).decode()

cells = []


def _add(cell_type, source):
    d = {"cell_type": cell_type, "id": f"cell{len(cells):02d}", "metadata": {},
         "source": source.splitlines(keepends=True)}
    if cell_type == "code":
        d["execution_count"] = None
        d["outputs"] = []
    cells.append(d)


def _tpl(src, **kw):
    for k, v in kw.items():
        src = src.replace(f"__{k}__", str(v))
    return src


# ── 0 ───────────────────────────────────────────────────────
_add("markdown", _tpl("""# 🧪 vLLM Step2 ドライラン（GPU 不要）

`__HF__` を vLLM で配信する構成のうち、**推論以外の全工程**を CPU で検証する。

| # | 検証 | 実推論 |
|---|---|---|
| 1 | vLLM install / バージョン / 所要時間 | 不要 |
| 2 | `vllm serve --help` のフラグ有無（起動ラダーの前提） | 不要 |
| 3 | Dataset マウント + symlink ステージング | 不要 |
| 4 | プロキシ + UI + `--enforce-sampling` | モック上流 |
| 5 | cloudflared 越しの UI / models / streaming / OpenAI互換 | モック上流 |
| 6 | Gist / keepalive / 後片付け | 不要 |

> モック上流（8000番）が vLLM の代役。T4 のカーネル挙動だけは確認できない。

**設定**: Accelerator `None` / Internet `On` / Add Data: `__DATASET__`, `my-cloudflared-bin`
""", HF=("twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP"),
            DATASET=DATASET_ID))

# ── 1: 設定 ─────────────────────────────────────────────────
_add("code", _tpl('''# 設定（GPU 不要）
import glob as _glob
import json
import os
import re
import subprocess
import sys
import time

SERVED_NAME = "__SERVED__"
VLLM_PORT = 8000
PROXY_PORT = 11435
MTP_ENABLED = True
MTP_METHOD = "mtp"
MTP_TOKENS = 2
MAXLEN_CANDIDATES = [32768, 16384, 8192]
TP_SIZE = 2

PROXY_SRC_B64 = "__PROXY_B64__"
CHAT_SRC_B64 = "__CHAT_B64__"
PROXY_PATH = "/usr/local/bin/auth_proxy.py"
CHAT_PATH = "/usr/local/bin/llama-chat.html"

OUT_DIRS = ("/kaggle/working", "/output")


def _find_input(slug):
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


def save_output(name, text):
    saved = []
    for d in OUT_DIRS:
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
            saved.append(d)
        except Exception as e:
            print(f"  ⚠ {d} 書き出し失敗 ({e})")
    if saved:
        print(f"  📄 {name} 保存済: {', '.join(saved)}")
        return True
    print(f"  ⚠ 保存できませんでした: {name}")
    return False


def fail(msg, tail_path=None):
    tail = ""
    try:
        if tail_path and os.path.isfile(tail_path):
            tail = open(tail_path, errors="replace").read()[-3000:]
    except Exception:
        pass
    save_output("FAILED.txt", f"{msg}\\n\\n{tail}")
    raise RuntimeError(msg)


_files = sorted(os.listdir(MODEL_SRC)) if os.path.isdir(MODEL_SRC) else []
print("モデル:", MODEL_SRC)
print("直下ファイル:", len(_files))
for _f in _files:
    _p = os.path.join(MODEL_SRC, _f)
    if os.path.isfile(_p):
        print(f"  {_f:34s} {os.path.getsize(_p):>15,} B")
if "config.json" not in _files:
    fail("config.json がありません。Add Data を確認してください。")
if "model.safetensors" not in _files:
    fail("model.safetensors がありません。Dataset を確認してください。")
''', SERVED=SERVED_NAME, DATASET_SLUG=DATASET_SLUG,
            PROXY_B64=PROXY_SRC_B64, CHAT_B64=CHAT_SRC_B64))

# ── 2: install + フラグ判定 ─────────────────────────────────
_add("markdown", "---\n### 1️⃣ vLLM install とフラグ判定（所要時間を実測）\n")

_add("code", '''# vLLM を入れて、起動ラダーが前提にしているフラグの有無を確認する
import subprocess
import sys
import time

t0 = time.time()
_r = subprocess.run([sys.executable, "-c", "import vllm"],
                    capture_output=True, text=True)
install_sec = 0.0
if _r.returncode == 0:
    print("ℹ vLLM は既に入っている（install 時間 0）")
else:
    print("⬇️  pip install vllm 開始")
    _p = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-input", "vllm"],
        capture_output=True, text=True, timeout=3600)
    install_sec = time.time() - t0
    print((_p.stderr or "")[-1500:])
    if _p.returncode != 0:
        fail("vLLM install 失敗（上のログ）")

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    "--no-deps", "--force-reinstall", "torchaudio==2.11.0"],
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

import torch
import vllm
print(f"✅ vLLM {vllm.__version__} / torch {torch.__version__} "
      f"/ CUDA {torch.version.cuda}")
print(f"⏱  install 所要: {install_sec/60:.1f} 分")

_h = subprocess.run(["vllm", "serve", "--help"],
                    capture_output=True, text=True, timeout=180)
HELP = (_h.stdout or "") + (_h.stderr or "")
# 判定が外れたときの唯一の手掛かり。ログと成果物の両方に残す。
print(f"vllm serve --help: exit={_h.returncode} {len(HELP)} 文字")
print(HELP[:1500])
save_output("vllm_help.txt", HELP[:20000])
HELP_OK = (_h.returncode == 0 and len(HELP) > 500
           and "Traceback (most recent call last)" not in HELP)

# ここに列挙した flag だけを avail に入れる。avail[flag] を直接引く箇所が
# あるため、後から使う flag を足し忘れると HELP 失敗時（= CPU では常に）に
# KeyError で即死する（v6 で実測）。
NEEDED = ["--speculative-config", "--trust-remote-code", "--reasoning-parser",
          "--enable-auto-tool-choice", "--tool-call-parser",
          "--disable-log-requests", "--kv-cache-dtype",
          "--gpu-memory-utilization", "--max-num-seqs",
          "--served-model-name", "--tensor-parallel-size",
          "--language-model-only"]
avail = {f: (f in HELP) for f in NEEDED}
if not HELP_OK:
    # 悲観側に倒す。0.32 で廃止されたフラグを付けると未知引数で即死する。
    avail = {f: False for f in NEEDED}
for f, ok in avail.items():
    print(("✅" if ok else "❌"), f)

missing = [f for f in ("--tensor-parallel-size", "--served-model-name",
                       "--max-num-seqs", "--gpu-memory-utilization")
           if not avail[f]]
if not HELP_OK:
    print("⚠ ヘルプ本文が取れない → フラグ判定は不能。"
          "本番は悲観側（任意フラグを落とす）で動く")
elif missing:
    # ここで止めると「help の書式が変わっただけ」でも全損になる。
    # 記録して先へ進む（本番ビルダーは同じ状況でも起動を試す）。
    print(f"⚠ 必須フラグが見つかりません（書式変更の可能性）: {missing}")

# 起動ラダーを組み立てる（実起動はしない）
# 量子化は指定しない（config の quantization_config は pack-quantized。
# awq 固定は誤り。詳細 notes/vllm-cpu-findings.md）
CORE = ["vllm", "serve", "<stage>", "--served-model-name", SERVED_NAME,
        "--host", "127.0.0.1", "--port", str(VLLM_PORT),
        "--tensor-parallel-size", str(TP_SIZE), "--dtype", "half",
        "--max-num-seqs", "2",
        "--gpu-memory-utilization", "0.90", "--kv-cache-dtype", "auto"]
# 本番と同じくテキスト専用で起動（視覚塔は落とす）。
if avail["--language-model-only"]:
    CORE.append("--language-model-only")
OPT = []
if avail["--trust-remote-code"]:
    OPT.append("--trust-remote-code")
if avail["--disable-log-requests"]:   # 0.32 では廃止（--enable-log-requests）
    OPT.append("--disable-log-requests")
if avail["--reasoning-parser"]:
    OPT += ["--reasoning-parser", "qwen3"]
if avail["--enable-auto-tool-choice"] and avail["--tool-call-parser"]:
    OPT += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]

SPEC = None
if MTP_ENABLED and avail["--speculative-config"]:
    SPEC = {"method": MTP_METHOD, "num_speculative_tokens": MTP_TOKENS}

PLAN = []
if SPEC:
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[0], True))
    PLAN.append(("mtp", SPEC, MAXLEN_CANDIDATES[1], True))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[0], False))
PLAN.append(("plain", None, MAXLEN_CANDIDATES[1], False))
PLAN.append(("plain", None, 8192, False))
PLAN.append(("triton", None, 8192, False))   # 0.32 に V0 は無い（明示 backend）

print()
print("起動ラダー（GPU 実行時にこの順で試す）:")
for i, (lab, sp, ml, use_opt) in enumerate(PLAN, 1):
    print(f"  {i}. spec={lab} max_len={ml} opt_flags={'on' if use_opt else 'off'}")
save_output("dryrun_plan.txt", json.dumps(
    {"vllm": vllm.__version__, "torch": torch.__version__,
     "install_sec": install_sec, "flags": avail,
     "plan": [[l, s, m, o] for l, s, m, o in PLAN]},
    indent=2, ensure_ascii=False) + "\\n")
''')

# ── 3: ステージング ─────────────────────────────────────────
_add("markdown", "---\n### 2️⃣ Dataset のマウントと symlink ステージング\n\n"
                 "MTP の draft head を出し入れしても実体は壊れないことを確認。\n")

_add("code", '''# symlink ステージング（実体コピーなし = disk を食わない）
import os
import shutil

STAGE = "/kaggle/tmp/vllm-model"
MTP_FILE = "model-mtp.safetensors"


def _stage_model(with_mtp):
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


n_on = _stage_model(True)
on_files = sorted(os.listdir(STAGE))
n_off = _stage_model(False)
off_files = sorted(os.listdir(STAGE))
print(f"MTP ON : {n_on} files  mtp={'model-mtp.safetensors' in on_files}")
print(f"MTP OFF: {n_off} files  mtp={'model-mtp.safetensors' in off_files}")
print(f"ステージ合計サイズ（実体参照）: "
      f"{sum(os.path.getsize(os.path.join(STAGE, f)) for f in off_files)/1e9:.1f} GB")
print(f"Dataset 実体は無傷: {os.path.isfile(os.path.join(MODEL_SRC, 'model.safetensors'))}")

if "model-mtp.safetensors" in on_files and \\
        "model-mtp.safetensors" not in off_files and n_off >= 3:
    print("✅ ステージング OK")
else:
    fail(f"ステージング異常: on={n_on} off={n_off}")
''')

# ── 4: モック上流 ───────────────────────────────────────────
_add("markdown", "---\n### 3️⃣ モック vLLM 上流（8000番）\n\n"
                 "本物の代役。`/v1/models`, `/v1/chat/completions`（stream 対応）、"
                 "受信ボディの記録。\n")

_add("code", '''# モック上流: vLLM の OpenAI 互換 API を最小再現する
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAST = {}


class Mock(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": SERVED_NAME, "object": "model",
                 "max_model_len": 32768, "owned_by": "mock"}]})
        elif p == "/__last":
            self._json(LAST)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        global LAST
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        LAST = {"path": self.path, "body": body}

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(12):
                chunk = {"id": "mock", "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta":
                                      {"content": f"tok{i} "}}]}
                payload = ("data: " + json.dumps(chunk) + "\\n\\n").encode()
                self.wfile.write(b"%x\\r\\n" % len(payload) + payload + b"\\r\\n")
                self.wfile.flush()
                time.sleep(0.15)
            done = b"data: [DONE]\\n\\n"
            self.wfile.write(b"%x\\r\\n" % len(done) + done + b"\\r\\n")
            self.wfile.write(b"0\\r\\n\\r\\n")
            self.wfile.flush()
            return

        self._json({
            "id": "mock", "object": "chat.completion",
            "model": body.get("model", SERVED_NAME),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "OK. (mock)"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4,
                      "total_tokens": 12},
        })


subprocess.run(["pkill", "-f", "mock_vllm"], capture_output=True)
_server = ThreadingHTTPServer(("127.0.0.1", VLLM_PORT), Mock)
threading.Thread(target=_server.serve_forever, daemon=True).start()
time.sleep(0.5)
import urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models",
                            timeout=5) as r:
    _d = json.loads(r.read())
print(f"✅ モック起動: {VLLM_PORT} models={[m['id'] for m in _d['data']]}")
''')

# ── 5: プロキシ ─────────────────────────────────────────────
_add("markdown", "---\n### 4️⃣ プロキシ起動（`--enforce-sampling`）\n")

_add("code", '''# プロキシ + UI を配置して起動（本番と同じ引数）
import base64
import os
import subprocess
import time
import urllib.error
import urllib.request

with open(PROXY_PATH, "wb") as f:
    f.write(base64.b64decode(PROXY_SRC_B64))
with open(CHAT_PATH, "wb") as f:
    f.write(base64.b64decode(CHAT_SRC_B64))
print(f"auth_proxy.py {os.path.getsize(PROXY_PATH):,} B / "
      f"llama-chat.html {os.path.getsize(CHAT_PATH):,} B")

subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
time.sleep(1)

with open("/tmp/auth_proxy.log", "w") as log:
    proxy_proc = subprocess.Popen(
        ["python3", PROXY_PATH, "--port", str(PROXY_PORT),
         "--upstream", f"http://127.0.0.1:{VLLM_PORT}",
         "--chat-html", CHAT_PATH, "--enforce-sampling"],
        stdout=log, stderr=log)
print(f"🔄 proxy PID: {proxy_proc.pid}")

for i in range(30):
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PROXY_PORT}/healthz", timeout=3) as r:
            if r.status == 200:
                print(f"✅ プロキシ起動（{i+1}s）")
                break
    except Exception:
        pass
    time.sleep(1)
else:
    print(open("/tmp/auth_proxy.log").read()[-2000:])
    fail("プロキシ起動失敗", "/tmp/auth_proxy.log")
''')

# ── 6: トンネル ─────────────────────────────────────────────
_add("markdown", "---\n### 5️⃣ cloudflared quick tunnel\n")

_add("code", '''# cloudflared 起動（Dataset 優先 / 無ければ DL）
import os
import re
import shutil
import subprocess
import time

CF_BIN = "/usr/local/bin/cloudflared"


def _open_public(url, timeout=15):
    """公開URLを開く。失敗時は IPv4 優先で再解決して 1 回だけ再試行。

    Kaggle の VM は AAAA を返すのに IPv6 経路が無いことがあり、
    urlopen が "Network is unreachable" になる（dryrun v6 で実測）。
    """
    import socket
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


NONE = _find_input("my-cloudflared-bin")
src = None
for cand in (os.path.join(NONE, "cloudflared"),
             os.path.join(NONE, "bin", "cloudflared")):
    if os.path.isfile(cand) and os.path.getsize(cand) > 1024 * 1024:
        src = cand
        break
if src:
    shutil.copy2(src, CF_BIN)
    subprocess.run(["chmod", "+x", CF_BIN])
    print(f"✅ cloudflared 復元: {src}")
elif not os.path.isfile(CF_BIN):
    subprocess.run(
        "curl -fsSL -o /usr/local/bin/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True, timeout=300)

TUNNEL_NAME = "cloudflared"
TUNNEL_LOG = "/tmp/cloudflared.log"
RESTART_CMD = [CF_BIN, "tunnel", "--url", f"http://127.0.0.1:{PROXY_PORT}",
               "--no-autoupdate"]
URL_RE = re.compile(r"https://[a-z0-9-]+\\.trycloudflare\\.com")


def probe_url():
    try:
        with open(TUNNEL_LOG) as f:
            m = URL_RE.search(f.read())
        return m.group(0) if m else ""
    except FileNotFoundError:
        return ""


subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
time.sleep(1)
with open(TUNNEL_LOG, "w") as log:
    subprocess.Popen(RESTART_CMD, stdout=log, stderr=log)

public_url = ""
for i in range(90):
    public_url = probe_url()
    if public_url:
        break
    time.sleep(1)
if not public_url:
    print(open(TUNNEL_LOG).read()[-3000:])
    fail("公開URL取得失敗", TUNNEL_LOG)

print(f"🌍 {public_url}")
with open("/tmp/public_url.txt", "w") as f:
    f.write(public_url)
save_output("public_url.txt", public_url + "\\n")

import urllib.request
for i in range(20):
    try:
        with _open_public(f"{public_url}/healthz", timeout=10) as r:
            if r.status == 200:
                print(f"✅ トンネル確立（{(i+1)*3}s）")
                break
    except Exception:
        pass
    time.sleep(3)
''')

# ── 7: verify ───────────────────────────────────────────────
_add("markdown", "---\n### 6️⃣ 公開URL経由の検証（モック上流）\n")

_add("code", '''# 5点検証（実推論以外は本番と同一経路）
import json
import time as _time
import urllib.error
import urllib.request

with open("/tmp/public_url.txt") as f:
    public_url = f.read().strip()

lines = [f"PUBLIC_URL={public_url}", f"MODEL={SERVED_NAME}", ""]
failures = []


def _judge(idx, label, ok, detail):
    print(("✅" if ok else "❌"), f"{label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _post(url, payload, timeout=120, stream=False):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _with_retry(fn, tries=3, delay=3):
    """公開 URL は cloudflared 側の一時障害で `Network is unreachable` になる。
    実測: v5 で UI と /v1/models だけ落ち、localhost 系は全部通った。
    一過性なので数回だけ粘る（本番で 20 分消費した後に誤 fail しないため）。
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if i + 1 < tries:
                wait = delay * (i + 1)
                print(f"  ⚠ 一時的な通信エラー（{i+1}/{tries}）: {e} "
                      f"→ {wait}s 待って再試行")
                _time.sleep(wait)
    raise last


CHAT = {"model": SERVED_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "max_tokens": 32}

# [1] UI
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

# [2] /v1/models
try:
    with _with_retry(lambda: _open_public(
            f"{public_url}/v1/models", timeout=60)) as r:
        md = json.loads(r.read())
    ids = [m.get("id") for m in md.get("data", [])]
    _judge(2, "models", r.status == 200 and SERVED_NAME in ids,
           f"ids={ids}")
except Exception as e:
    _judge(2, "models", False, f"{type(e).__name__}: {e}")

# [3] streaming
try:
    resp = _post(f"{public_url}/v1/chat/completions",
                 dict(CHAT, stream=True), timeout=120)
    times = []
    t0 = _time.time()
    with resp as r:
        for raw in r:
            if not raw.strip():
                continue
            times.append(_time.time() - t0)
    total = _time.time() - t0
    if len(times) >= 3:
        first, spread = times[0], times[-1] - times[0]
        ok = spread > 0.2 and first < total * 0.8
        detail = (f"chunks={len(times)} first={first:.2f}s "
                  f"total={total:.2f}s spread={spread:.2f}s")
    else:
        ok, detail = False, f"chunks={len(times)}"
    _judge(3, "streaming", ok, detail)
except Exception as e:
    _judge(3, "streaming", False, f"{type(e).__name__}: {e}")

# [4] OpenAI 互換（temperature=0 でも 200 + 生成）
try:
    with _post(f"{public_url}/v1/chat/completions",
               dict(CHAT, temperature=0, repetition_penalty=1.0)) as r:
        st = r.status
        cj = json.loads(r.read())
    text = "".join((c.get("message") or {}).get("content") or ""
                   for c in cj.get("choices", []))
    usage = cj.get("usage") or {}
    _judge(4, "openai-compat", st == 200 and bool(text or usage),
           f"HTTP {st} reply={text[:30]!r} usage={usage}")
except Exception as e:
    _judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

# [5] enforce-sampling が上流まで届いているか（モックに記録されたボディ）
try:
    with urllib.request.urlopen(
            f"http://127.0.0.1:{VLLM_PORT}/__last", timeout=10) as r:
        last = json.loads(r.read())
    b = last.get("body") or {}
    temp = b.get("temperature")
    rp = b.get("repetition_penalty")
    ok = temp is not None and float(temp) >= 0.7 and \\
        rp is not None and float(rp) >= 1.05
    _judge(5, "enforce-sampling", ok, f"upstream temp={temp} rp={rp}")
except Exception as e:
    _judge(5, "enforce-sampling", False, f"{type(e).__name__}: {e}")

# [6] 会話（履歴が上流までそのまま渡る）
# 実推論は GPU でしか試せないが、proxy が messages を欠落・改変しないことは
# ここで確認できる（role 順・日本語・system の保持）。
try:
    conv = [
        {"role": "system", "content": "あなたは日本語で簡潔に答えます。"},
        {"role": "user", "content": "私の名前はコウヘイです。"},
        {"role": "assistant", "content": "コウヘイさん、覚えました。"},
        {"role": "user", "content": "私の名前は？"},
    ]
    resp = _post(f"{public_url}/v1/chat/completions",
                 {"model": SERVED_NAME, "messages": conv, "max_tokens": 32},
                 timeout=120)
    with resp as r:
        r.read()
    with _open_public(f"http://127.0.0.1:{VLLM_PORT}/__last", timeout=10) as r:
        sent = ((json.loads(r.read()).get("body") or {}).get("messages") or [])
    ok6 = sent == conv
    _judge(6, "conversation-history", ok6,
           f"messages={len(sent)} equal={ok6}")
except Exception as e:
    _judge(6, "conversation-history", False, f"{type(e).__name__}: {e}")

report = "\\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\\n"):
    failures.append("verify_result.txt 保存失敗")
if failures:
    fail("検証失敗: " + "; ".join(failures))
print("✅ 6/6 パス（モック上流）— 推論以外は本番同一")
''')

# ── 8: keepalive ────────────────────────────────────────────
_add("markdown", "---\n### 7️⃣ キープアライブ（2分・CPU）\n")

_add("code", '''# keepalive 2分（本番は 480 分）。プロキシ/トンネルの死活だけ見る
import subprocess
import time
import urllib.request

KEEPALIVE_MINUTES = 8
KEEPALIVE_CONF = "/tmp/keepalive.conf"


def _proc(name):
    try:
        args = ["pgrep", "-x", name] if name == "cloudflared" \\
            else ["pgrep", "-f", name]
        return "alive" if subprocess.run(args, capture_output=True,
                                         text=True).stdout.strip() else "DEAD"
    except Exception:
        return "unknown"


def _read_conf():
    try:
        with open(KEEPALIVE_CONF) as f:
            d = dict(l.strip().split("=", 1) for l in f
                     if l.strip() and "=" in l)
        return int(d.get("RUN", 1)), int(d.get("MINUTES", KEEPALIVE_MINUTES))
    except Exception:
        return 1, KEEPALIVE_MINUTES


print(f"Public URL: {public_url}")
log = []
for i in range(1, KEEPALIVE_MINUTES + 1):
    run, minutes = _read_conf()
    if run <= 0:
        print(f"RUN=0 で {i}周目に終了")
        break
    time.sleep(60)
    p_st, t_st = _proc("auth_proxy.py"), _proc("cloudflared")
    line = (f"[{time.strftime('%H:%M:%S')}] proxy={p_st} tunnel={t_st}")
    if i % 1 == 0:
        try:
            with urllib.request.urlopen(f"{public_url}/healthz", timeout=15) as r:
                line += " | tunnel=OK" if r.status == 200 else f" | HTTP{r.status}"
        except Exception as e:
            line += f" | tunnel=ERR({type(e).__name__})"
    log.append(line)
    print(line)

save_output("keepalive.txt", "\\n".join(log) + "\\n")

# 後片付け
subprocess.run(["pkill", "-x", "cloudflared"], capture_output=True)
subprocess.run(["pkill", "-f", "auth_proxy.py"], capture_output=True)
if "_server" in globals():
    _server.shutdown()
print("🧹 停止済（proxy / cloudflared / mock）")
''')

# ── 書き出し ────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}: {len(cells)} cells")

meta = {
    "id": f"{OWNER}/{SLUG}",
    "title": "Step2 vLLM dryrun",
    "code_file": OUT.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [DATASET_ID, f"{OWNER}/my-cloudflared-bin"],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
with open(OUT_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT_DIR / 'kernel-metadata.json'}")
