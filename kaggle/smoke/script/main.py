#!/usr/bin/env python3
"""smoke orcarouter qwen38 27b script（自動生成。直接編集しない）。

生成: kaggle/build_batch_script.py ← kaggle/build_smoke.py のセル列
"""


# ── 🧪 Smoke test: orcarouter/Qwen3.8-27B-Uncensored:latest on T4 x2 


# ── 🧩 共通ランタイム（GPU / chat） ───────────────────────


# ============================================================
# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）
# proxy / tunnel / keepalive / verify の配管をここに集約する。
# ============================================================
# ── runtime/proc.py ───────────────────────────────────────
"""プロセス・GPU 状態の確認と後片付け（生成コードへ埋め込む）。"""
import re
import subprocess
import time


def _kc_live(pids, timeout=5):
    """zombie を除いた PID を返す。kill 直後は未 reap の Z が pgrep に残る。"""
    if not pids:
        return []
    try:
        out = subprocess.run(
            ["ps", "-o", "pid=,stat=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except Exception:
        return pids
    live = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not parts[1].startswith("Z"):
            live.append(parts[0])
    return live


def kc_proc_state(name, match="exact", timeout=5):
    """alive / DEAD / unknown。match=exact は pgrep -x、argv は pgrep -f。"""
    try:
        flag = "-x" if match == "exact" else "-f"
        out = subprocess.run(["pgrep", flag, name], capture_output=True,
                             text=True, timeout=timeout).stdout
        pids = [p for p in out.split() if p.strip()]
        return "alive" if _kc_live(pids, timeout) else "DEAD"
    except Exception:
        return "unknown"


def kc_gpu_usage(timeout=5):
    """nvidia-smi の使用率/VRAM を 1 行で返す（CPU 環境では N/A）。"""
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip().replace("\n", " | ")
    except Exception:
        return "N/A"


def kc_context_budget(vram_total_mib, model_gib, kv_kib_per_token, candidates,
                      requested=None, budget_ratio=0.88):
    """VRAM から (コンテキスト長, 予算 GiB) を決める。

    予算は実効 VRAM の `budget_ratio` 倍（T4 x2 の 30 GiB をそのまま
    使うと CUDA graph / activation で溢れるため 0.88 で見積もる）。
    `requested` が収まるならそれを優先し、駄目なら `candidates` を
    大きい順に試す。どれも収まらなければ最小候補を返す（起動は試す）。
    """
    budget_gib = vram_total_mib / 1024 * budget_ratio

    def fits(ctx):
        return model_gib + kv_kib_per_token * ctx / 1024 / 1024 <= budget_gib

    if requested and fits(requested):
        return requested, budget_gib
    for ctx in candidates:
        if fits(ctx):
            return ctx, budget_gib
    return candidates[-1], budget_gib


def kc_context_gib(model_gib, kv_kib_per_token, ctx):
    """コンテキスト 1 本ぶんの見積り（モデル + KV）を GiB で返す。"""
    return model_gib + kv_kib_per_token * ctx / 1024 / 1024


def kc_detect_gpu(timeout=15):
    """全 GPU の (名前, VRAM MiB) を返す。検出失敗は ([], [])。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip()
        names, mems = [], []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            names.append(parts[0].strip())
            mems.append(int(re.sub(r"[^0-9]", "", parts[1])))
        return names, mems
    except Exception as e:
        print(f"GPU 検出失敗: {e}")
        return [], []


def kc_wait_gpu_idle(limit_mib=500, timeout=180):
    """全 GPU の使用量が limit 未満になるまで待つ。経過秒、失敗は -1。

    名前は ollama の kc_wait_vram_free（/api/ps を見る）と衝突させない。
    """
    for i in range(timeout):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        used = [int(x) for x in out.splitlines() if x.strip().isdigit()]
        if used and max(used) < limit_mib:
            return i
        time.sleep(1)
    return -1


def _kc_argv(pattern):
    """"-x ngrok" → ["-x", "ngrok"]（shell を介さない）。"""
    return pattern.split()


def kc_pids(pattern, timeout=15):
    """pgrep の引数文字列（例: "-x ngrok"）に一致する PID を返す。"""
    try:
        r = subprocess.run(["pgrep", *_kc_argv(pattern)],
                           capture_output=True, text=True, timeout=timeout)
        return _kc_live([p for p in r.stdout.split() if p.strip()], timeout)
    except Exception:
        return []


def kc_stop_processes(targets, term_wait=3, final_wait=2):
    """(pgrep_pattern, label) を TERM → 待機 → KILL で止め、残存ラベルを返す。"""
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-TERM", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  TERM → {label}")
    time.sleep(term_wait)
    for pattern, label in targets:
        if kc_pids(pattern):
            subprocess.run(["pkill", "-KILL", *_kc_argv(pattern)],
                           capture_output=True)
            print(f"  KILL → {label}（残存していた）")
    time.sleep(final_wait)
    return [label for pattern, label in targets if kc_pids(pattern)]

# ── runtime/engines/ollama.py ─────────────────────────────
"""Ollama 固有の推論・モデル解放処理（生成コードへ埋め込む）。"""
import json
import time
import urllib.request


def kc_ollama_chat(port, model, prompt, num_predict=64, timeout=1800,
                   think=None):
    """Ollama /api/chat を 1 回叩き、応答 dict を返す（非ストリーム）。

    think=None はボディにキーを入れない（Ollama 既定に任せる）。
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def kc_unload_model(port, model, timeout=60):
    """keep_alive=0 でモデルを VRAM から降ろす。1 経路でも通れば True。

    プロセスを殺すより先に、Ollama 自身に VRAM を返させる。
    """
    unloaded = False
    for path, payload in [
        ("/api/generate", {"model": model, "keep_alive": 0}),
        ("/api/chat", {"model": model, "messages": [], "keep_alive": 0}),
    ]:
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 200:
                    unloaded = True
                    print(f"  ✅ アンロード要求 OK ({path})")
                    break
        except Exception as e:
            print(f"  —  {path}: {type(e).__name__}")
    return unloaded


def kc_wait_vram_free(port, attempts=10, interval=3):
    """常駐モデルが 0 になるまで待つ。空なら True。"""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/ps", timeout=10) as r:
                models = json.loads(r.read())
            loaded = models.get("models") or []
            if not loaded:
                print("  ✅ VRAM 解放確認（常駐モデル 0）")
                return True
            vram = sum(m.get("size_vram", 0) for m in loaded)
            print(f"  ⏳ まだ常駐: {vram:,} B（{attempt + 1}/{attempts}）")
        except Exception as e:
            print(f"  —  /api/ps 確認失敗: {type(e).__name__}")
            return False
        time.sleep(interval)
    print("  ⚠ アンロードを確認できず（プロセス kill へ進む）")
    return False


# ── 1️⃣ GPU 確認 ──────────────────────────────────


import subprocess

GPU_NAMES, VRAM_EACH = kc_detect_gpu()
VRAM_TOTAL = sum(VRAM_EACH)

print(f"GPU  : {GPU_NAMES}")
print(f"VRAM : {VRAM_EACH} MiB each  →  合計 {VRAM_TOTAL} MiB ({VRAM_TOTAL/1024:.1f} GiB)")
print()
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

if VRAM_TOTAL == 0:
    raise RuntimeError("GPU なし。Accelerator を 'GPU T4 x2' に設定してください。")


# ── 2️⃣ Ollama インストール ───────────────────────────


import subprocess, os

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                   shell=True, capture_output=True, text=True, timeout=900)
print(r.stdout[-1500:])
print(r.stderr[-1500:])

for p in ["/usr/local/bin/ollama", "/usr/local/lib/ollama/llama-server"]:
    ok = os.path.isfile(p)
    print(f"{'✅' if ok else '❌'} {p}  ({os.path.getsize(p) if ok else 0:,} B)")

print(subprocess.run(["/usr/local/bin/ollama", "--version"],
                     capture_output=True, text=True).stdout.strip())


# ── 3️⃣ コンテキスト長を VRAM から決定 ──────────────────────


# 実測値（GGUF ヘッダより）
#   full attention 17層 × kv_head 4 × (key 256 + value 256) × 2B = 68 KiB/token (f16)
#   q8_0 KV ならその半分 = 34 KiB/token
MODEL_GIB = 16.52
KV_KIB_PER_TOK_Q8 = 34.0

budget_gib = VRAM_TOTAL / 1024 * 0.88
print(f"予算: {budget_gib:.1f} GiB（実効 VRAM の 88%）")
print()
print(f"{'ctx':>8} | {'KV(q8_0)':>9} | {'合計':>8} | 判定")
print("-" * 44)

CONTEXT_LENGTH = 8192
for c in [262144, 131072, 65536, 32768, 16384, 8192]:
    kv = KV_KIB_PER_TOK_Q8 * c / 1024 / 1024
    total = MODEL_GIB + kv
    fits = total <= budget_gib
    if fits and CONTEXT_LENGTH == 8192:
        CONTEXT_LENGTH = c
    print(f"{c:>8} | {kv:>7.2f}GiB | {total:>6.2f}GiB | {'✅' if fits else '❌'}")

print()
print(f"→ 採用: {CONTEXT_LENGTH:,} tokens")


# ── 4️⃣ Ollama 起動 + モデル pull（16.5 GiB / 10〜30分） ─


import os, subprocess, time

MODEL_NAME = "orcarouter/Qwen3.8-27B-Uncensored:latest"
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
os.makedirs(OLLAMA_MODELS, exist_ok=True)

env = {
    **os.environ,
    "OLLAMA_MODELS": OLLAMA_MODELS,
    "OLLAMA_HOST": "0.0.0.0:11434",
    "OLLAMA_ORIGINS": "*",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NUM_PARALLEL": "1",
}

subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
time.sleep(2)

OLLAMA_BIN = "/usr/local/bin/ollama"
with open("/tmp/ollama.log", "w") as log:
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)
print(f"🔄 Ollama PID: {proc.pid}")

for i in range(60):
    if subprocess.run([OLLAMA_BIN, "list"], env=env,
                      capture_output=True).returncode == 0:
        print(f"✅ サーバー起動（{i+1}s）")
        break
    time.sleep(1)
else:
    print(open("/tmp/ollama.log").read()[-3000:])
    raise RuntimeError("Ollama 起動失敗")

print(f"⏳ pull: {MODEL_NAME}")
t0 = time.time()
r = subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=env,
                   capture_output=True, text=True, timeout=7200)
print(r.stdout[-800:])
print(r.stderr[-800:])
print(f"✅ 完了（{(time.time()-t0)/60:.1f} 分）")
print()
print(subprocess.run([OLLAMA_BIN, "list"], env=env,
                     capture_output=True, text=True).stdout)


# ── 5️⃣ 推論テスト（本番） ───────────────────────────────


import time

def _chat(prompt, num_predict=64, timeout=1800):
    return kc_ollama_chat(11434, MODEL_NAME, prompt,
                          num_predict=num_predict, timeout=timeout)

print("=" * 60)
print("  テスト1: 短文（モデルロード込み）")
print("=" * 60)
t0 = time.time()
try:
    d = _chat("Say OK.", num_predict=16)
    dt = time.time() - t0
    print(f"✅ 成功（{dt:.1f}s）")
    print(f"   load_duration : {(d.get('load_duration') or 0)/1e9:.1f}s")
    print(f"   reply         : {d.get('message', {}).get('content', '')!r}")
    ec, ed = d.get("eval_count") or 0, (d.get("eval_duration") or 1) / 1e9
    print(f"   速度          : {ec/ed:.1f} tokens/s" if ec else "   速度: N/A")
except Exception as e:
    print(f"❌ 失敗: {e}")
    print(open("/tmp/ollama.log").read()[-4000:])

print()
print("=" * 60)
print("  テスト2: 日本語 + 思考（実用チェック）")
print("=" * 60)
t0 = time.time()
try:
    d = _chat("1から10までの素数を列挙して。簡潔に。", num_predict=256)
    dt = time.time() - t0
    print(f"✅ 成功（{dt:.1f}s）")
    print(f"   reply: {d.get('message', {}).get('content', '')}")
    th = d.get("message", {}).get("thinking") or ""
    if th:
        print(f"   thinking ({len(th)} chars): {th[:200]}...")
    ec, ed = d.get("eval_count") or 0, (d.get("eval_duration") or 1) / 1e9
    print(f"   速度: {ec/ed:.1f} tokens/s" if ec else "   速度: N/A")
except Exception as e:
    print(f"❌ 失敗: {e}")

print()
print("=" * 60)
print("  VRAM 使用状況")
print("=" * 60)
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
     "--format=csv"], capture_output=True, text=True).stdout)


# ── 6️⃣ 結果 ──────────────────────────────────────
