#!/usr/bin/env python3
"""Smoke test: prove Ollama + orcarouter/Qwen3.8-27B-Uncensored runs on Kaggle T4 x2.
No Kaggle Secrets required. No Dataset required (pulls from registry).
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kbuild import pack_nb  # noqa: E402

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
# 出力先は __file__ 基準（cwd 依存だと kaggle/ 直下などに別物ができる）
OUT = str(HERE / "smoke" / "smoke-orcarouter-qwen38-27b.ipynb")

cells = []

cells.append(("markdown", f"""# 🧪 Smoke test: {MODEL} on T4 x2

**Secrets 不要 / Dataset 不要。** このノートブックだけで完結します。

### 目的
1. T4 x2 が認識されるか
2. Ollama が入るか
3. 16.5 GiB のモデルが VRAM に載るか
4. 推論が通るか（tokens/s はいくつか）

### 設定
- Accelerator: `GPU T4 x2`
- Internet: `On`

> 公開（cloudflared）はしません。まずスタックが動くことを確認するのが目的です。
"""))

cells.append(("markdown", "---\n### 1️⃣ GPU 確認\n"))

cells.append(("code", '''import os, re, subprocess

def _detect_gpu():
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
            p = line.split(",")
            names.append(p[0].strip())
            mems.append(int(re.sub(r"[^0-9]", "", p[1])))
        return names, mems
    except Exception as e:
        print(f"検出失敗: {e}")
        return [], []

GPU_NAMES, VRAM_EACH = _detect_gpu()
VRAM_TOTAL = sum(VRAM_EACH)

print(f"GPU  : {GPU_NAMES}")
print(f"VRAM : {VRAM_EACH} MiB each  →  合計 {VRAM_TOTAL} MiB ({VRAM_TOTAL/1024:.1f} GiB)")
print()
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

if VRAM_TOTAL == 0:
    raise RuntimeError("GPU なし。Accelerator を 'GPU T4 x2' に設定してください。")
'''))

cells.append(("markdown", "---\n### 2️⃣ Ollama インストール\n"))

cells.append(("code", '''import subprocess, os

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
'''))

cells.append(("markdown", "---\n### 3️⃣ コンテキスト長を VRAM から決定\n"))

cells.append(("code", '''# 実測値（GGUF ヘッダより）
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
'''))

cells.append(("markdown", "---\n### 4️⃣ Ollama 起動 + モデル pull（16.5 GiB / 10〜30分）\n"))

cells.append(("code", f'''import os, subprocess, time

MODEL_NAME = "{MODEL}"
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
os.makedirs(OLLAMA_MODELS, exist_ok=True)

env = {{
    **os.environ,
    "OLLAMA_MODELS": OLLAMA_MODELS,
    "OLLAMA_HOST": "0.0.0.0:11434",
    "OLLAMA_ORIGINS": "*",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NUM_PARALLEL": "1",
}}

subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
time.sleep(2)

OLLAMA_BIN = "/usr/local/bin/ollama"
with open("/tmp/ollama.log", "w") as log:
    proc = subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)
print(f"🔄 Ollama PID: {{proc.pid}}")

for i in range(60):
    if subprocess.run([OLLAMA_BIN, "list"], env=env,
                      capture_output=True).returncode == 0:
        print(f"✅ サーバー起動（{{i+1}}s）")
        break
    time.sleep(1)
else:
    print(open("/tmp/ollama.log").read()[-3000:])
    raise RuntimeError("Ollama 起動失敗")

print(f"⏳ pull: {{MODEL_NAME}}")
t0 = time.time()
r = subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=env,
                   capture_output=True, text=True, timeout=7200)
print(r.stdout[-800:])
print(r.stderr[-800:])
print(f"✅ 完了（{{(time.time()-t0)/60:.1f}} 分）")
print()
print(subprocess.run([OLLAMA_BIN, "list"], env=env,
                     capture_output=True, text=True).stdout)
'''))

cells.append(("markdown", "---\n### 5️⃣ 推論テスト（本番）\n"))

cells.append(("code", '''import json, time, urllib.request

def _chat(prompt, num_predict=64, timeout=1800):
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

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
'''))

cells.append(("markdown", """---
### 6️⃣ 結果

上のセルで **テスト1 / テスト2 が ✅ になれば、スタックは動いています。**

次に公開する場合は `step2-orcarouter-qwen38-27b-cloudflared.ipynb` を実行します
（cloudflared の quick tunnel。authtoken 不要）。
"""))

pack_nb.write(OUT, cells, cell_ids=True)
