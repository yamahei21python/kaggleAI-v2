#!/usr/bin/env python3
"""Benchmark: context length vs speed for orcarouter/Qwen3.8-27B-Uncensored on T4 x2"""
import pathlib
import json

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:latest"
# 出力先は __file__ 基準（cwd 依存だと kaggle/ 直下などに別物ができる）
OUT = str(pathlib.Path(__file__).resolve().parent / "bench" / "bench-ctx-orcarouter-qwen38-27b.ipynb")

cells = []

cells.append(("markdown", f"""# 📊 Benchmark: コンテキスト長 vs 速度

`{MODEL}` を T4 x2 で動かし、**コンテキスト長が生成速度に与える影響**を測ります。

### 仮説
前回の smoke test は 262,144 を自動選択し、**3.6 tokens/s** でした。
T4 x2 にしては遅い。262K の KV 確保が原因ではないか。

### 方法
`OLLAMA_CONTEXT_LENGTH` を変えて Ollama を再起動し、同じプロンプトで測ります。

> ⚠ 再起動時は `ollama` と `llama-server` の**両方**を落とし、VRAM が解放されたのを
> 確認してから次を測ります。`pkill ollama` だけでは子プロセスが残り、数値が汚染されます。

| ctx | KV(q8_0) | モデル+KV |
|---|---|---|
| 32,768 | 1.06 GiB | 17.58 GiB |
| 131,072 | 4.25 GiB | 20.77 GiB |
| 262,144 | 8.50 GiB | 25.02 GiB |

### 設定
- Accelerator: `GPU T4 x2` / Internet: `On`
- Secrets 不要
"""))

cells.append(("markdown", "---\n### 1️⃣ GPU 確認\n"))

cells.append(("code", '''import os, re, subprocess

def _gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        names, mems = [], []
        for line in out.splitlines():
            if line.strip():
                p = line.split(",")
                names.append(p[0].strip())
                mems.append(int(re.sub(r"[^0-9]", "", p[1])))
        return names, mems
    except Exception as e:
        print("検出失敗:", e)
        return [], []

GPU_NAMES, VRAM_EACH = _gpus()
VRAM_TOTAL = sum(VRAM_EACH)
print(f"GPU  : {GPU_NAMES}")
print(f"VRAM : {VRAM_EACH} MiB each → 合計 {VRAM_TOTAL} MiB ({VRAM_TOTAL/1024:.1f} GiB)")
if VRAM_TOTAL == 0:
    raise RuntimeError("GPU なし。Accelerator を 'GPU T4 x2' に設定してください。")
'''))

cells.append(("markdown", "---\n### 2️⃣ Ollama インストール + モデル pull\n"))

cells.append(("code", f'''import os, subprocess, time

MODEL_NAME = "{MODEL}"
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
os.makedirs(OLLAMA_MODELS, exist_ok=True)

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True, timeout=900)
print("✅ Ollama installed")

OLLAMA_BIN = "/usr/local/bin/ollama"

# pull は 1 回だけでよい（コンテキストは起動時に決まる）
base_env = {{**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS}}
with open("/tmp/ollama_boot.log", "w") as log:
    subprocess.Popen([OLLAMA_BIN, "serve"], env=base_env, stdout=log, stderr=log)
for i in range(60):
    if subprocess.run([OLLAMA_BIN, "list"], env=base_env,
                      capture_output=True).returncode == 0:
        print(f"✅ server up ({{i+1}}s)")
        break
    time.sleep(1)

print(f"⏳ pull: {{MODEL_NAME}}")
t0 = time.time()
subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=base_env,
               check=True, timeout=7200)
print(f"✅ pull 完了 ({{(time.time()-t0)/60:.1f}} 分)")
'''))

cells.append(("markdown", "---\n### 3️⃣ ベンチマーク本体\n"))

cells.append(("code", '''# 各コンテキスト長で Ollama を再起動 → 同じプロンプトで計測
import json, os, subprocess, time, urllib.request

OLLAMA_BIN = "/usr/local/bin/ollama"
PROMPT = "1から20までの素数を列挙して。数字だけをカンマ区切りで。"
CTX_LIST = [32768, 131072, 262144]


def _free_vram(timeout=180):
    """llama-server が VRAM を手放すまで待つ。戻り値は待った秒数（失敗は -1）。"""
    for i in range(timeout):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        used = [int(x) for x in out.splitlines() if x.strip().isdigit()]
        if used and max(used) < 500:
            return i
        time.sleep(1)
    return -1


def _restart(ctx):
    """指定コンテキストで Ollama を起動し直す。

    pkill ollama は名前完全一致なので llama-server 子プロセスは残る。
    残ると VRAM を掴んだまま次の測定が走り、数値が汚染される（実測で確認）。
    """
    subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
    subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
    time.sleep(4)

    freed = _free_vram()
    if freed < 0:
        raise RuntimeError("VRAM が解放されませんでした（前の llama-server が残留）")
    print(f"  VRAM 解放: {freed}s")

    env = {
        **os.environ,
        "OLLAMA_MODELS": OLLAMA_MODELS,
        "OLLAMA_HOST": "0.0.0.0:11434",
        "OLLAMA_ORIGINS": "*",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_CONTEXT_LENGTH": str(ctx),
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_NUM_PARALLEL": "1",
    }
    with open(f"/tmp/ollama_{ctx}.log", "w") as log:
        subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)

    for i in range(120):
        if subprocess.run([OLLAMA_BIN, "list"], env=env,
                          capture_output=True).returncode == 0:
            return env, i + 1
        time.sleep(1)
    raise RuntimeError(f"起動失敗 ctx={ctx}")


def _chat(prompt, num_predict=64, timeout=2400):
    body = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


results = []

for ctx in CTX_LIST:
    print("=" * 64)
    print(f"  ctx = {ctx:,}")
    print("=" * 64)

    try:
        env, boot_s = _restart(ctx)
        print(f"  起動: {boot_s}s")

        # --- 1回目: ロード + ウォームアップ（計測から除外）---
        t0 = time.time()
        warm = _chat("Say OK.", num_predict=8)
        warm_s = time.time() - t0
        load_s = (warm.get("load_duration") or 0) / 1e9
        print(f"  ウォームアップ: {warm_s:.1f}s (load {load_s:.1f}s)")
        print("  VRAM(ロード後): " + subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().replace(chr(10), " / "))

        # --- 2回目: 本計測 ---
        t0 = time.time()
        d = _chat(PROMPT, num_predict=64)
        wall = time.time() - t0

        ec = d.get("eval_count") or 0
        ed = (d.get("eval_duration") or 1) / 1e9
        pc = d.get("prompt_eval_count") or 0
        pd = (d.get("prompt_eval_duration") or 1) / 1e9

        row = {
            "ctx": ctx,
            "load_s": round(load_s, 1),
            "prompt_tokens": pc,
            "prompt_s": round(pd, 2),
            "prompt_tps": round(pc / pd, 1) if pd else None,
            "gen_tokens": ec,
            "gen_s": round(ed, 2),
            "gen_tps": round(ec / ed, 1) if ec else None,
            "wall_s": round(wall, 1),
            "reply": (d.get("message", {}).get("content") or "")[:80],
        }
        results.append(row)

        print(f"  生成: {ec} tokens / {ed:.1f}s → {row['gen_tps']} tokens/s")
        print(f"  プロンプト処理: {pc} tokens / {pd:.2f}s → {row['prompt_tps']} tokens/s")
        print(f"  reply: {row['reply']!r}")

        # --- VRAM ---
        print(subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())

    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        results.append({"ctx": ctx, "error": str(e)})

print()
print("=" * 64)
print("  結果まとめ")
print("=" * 64)
print(f"{'ctx':>9} | {'load':>6} | {'prompt tps':>10} | {'gen tps':>8} | {'wall':>7}")
print("-" * 56)
for r in results:
    if "error" in r:
        print(f"{r['ctx']:>9,} | {'—':>6} | {'—':>10} | {'—':>8} | ERROR")
    else:
        print(f"{r['ctx']:>9,} | {r['load_s']:>5.1f}s | {r['prompt_tps']:>10} | "
              f"{r['gen_tps']:>8} | {r['wall_s']:>6.1f}s")

print()
print("JSON:")
print(json.dumps(results, ensure_ascii=False, indent=2))
'''))

cells.append(("markdown", """---
### 4️⃣ 読み方

- **gen tps** が実用速度。これが ctx によってどう変わるかを見ます
- **prompt tps** は入力処理速度。長いコンテキストではここが効きます
- 差が出なければ、遅さの原因はコンテキストではなく **2 GPU 分割（PCIe 経由）** です
  - 16.52 GiB は T4 1枚（15 GiB）に載らないため、分割は避けられません
"""))

nb = {
    "cells": [
        {
            "cell_type": t,
            "id": f"cell-{i:02d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **(dict(execution_count=None, outputs=[]) if t == "code" else {}),
        }
        for i, (t, src) in enumerate(cells)
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
