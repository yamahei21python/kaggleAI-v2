#!/usr/bin/env python3
"""Bench: llama-server 直起動 + MTP 投機的デコード（--spec-type draft-mtp）の効果測定

Ollama を経由せず llama-server を直に起動し、MTP の有無で生成速度を比較する。
モデルは orcarouter/Qwen3.8-27B-Uncensored（qwen35、nextn_predict_layers=1 内蔵）。
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kbuild import pack_nb  # noqa: E402

# 出力先は __file__ 基準（cwd 依存だと kaggle/ 直下などに別物ができる）
OUT = str(HERE / "bench_mtp" / "bench-mtp-orcarouter-qwen38-27b.ipynb")

cells = []

cells.append(("markdown", """# 📊 Benchmark: llama-server 直起動 + MTP 投機的デコード

`orcarouter/Qwen3.8-27B-Uncensored` を **Ollama を経由せず llama-server で直に起動**し、
**MTP（Multi-Token Prediction）投機的デコード**の効果を測ります。

### 背景
- モデルの GGUF に `qwen35.nextn_predict_layers = 1` がある（MTP 層を内蔵）
- llama.cpp は `--spec-type draft-mtp` で MTP を使える
- Ollama はこれを環境変数で露出していない → **直起動が必要**

### 方法
同じプロンプトで `--spec-type` を `none` / `draft-mtp` と変えて比較します。

| 条件 | フラグ |
|---|---|
| ベースライン | `--spec-type none` |
| MTP | `--spec-type draft-mtp` |

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

cells.append(("markdown", "---\n### 2️⃣ llama-server バイナリの入手\n"))

cells.append(("code", '''# llama-server を単体で入手する。
# Ollama の install.sh は llama-server を同梱するので、それを使うのが最短。
import os, subprocess, time, glob

OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
os.makedirs(OLLAMA_MODELS, exist_ok=True)

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True, timeout=900)
print("✅ Ollama installed（llama-server 同梱）")

# llama-server の実体を探す
CANDIDATES = [
    "/usr/local/lib/ollama/llama-server",
    "/usr/lib/ollama/llama-server",
]
LLAMA_SERVER = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if not LLAMA_SERVER:
    hits = glob.glob("/usr/**/llama-server", recursive=True)
    LLAMA_SERVER = hits[0] if hits else None

if not LLAMA_SERVER:
    raise RuntimeError("llama-server が見つかりません")

print(f"✅ llama-server: {LLAMA_SERVER} ({os.path.getsize(LLAMA_SERVER):,} B)")
print()
print(subprocess.run([LLAMA_SERVER, "--version"],
                     capture_output=True, text=True).stdout.strip()[:300])

# MTP が使えるか（--spec-type に draft-mtp があるか）
help_txt = subprocess.run([LLAMA_SERVER, "--help"],
                          capture_output=True, text=True).stdout
HAS_MTP = "draft-mtp" in help_txt
print()
print(f"{'✅' if HAS_MTP else '❌'} --spec-type draft-mtp: {'あり' if HAS_MTP else 'なし'}")
if not HAS_MTP:
    print("  → この llama-server では MTP を検証できません")
'''))

cells.append(("markdown", "---\n### 3️⃣ モデル GGUF を入手\n"))

cells.append(("code", '''# Ollama で pull してから blob から GGUF を取り出す。
# （llama-server は GGUF を直接読むので、Ollama の blob をそのまま使える）
import os, subprocess, time, glob, json

MODEL_NAME = "orcarouter/Qwen3.8-27B-Uncensored:latest"
OLLAMA_BIN = "/usr/local/bin/ollama"

env = {**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS}
with open("/tmp/ollama_boot.log", "w") as log:
    subprocess.Popen([OLLAMA_BIN, "serve"], env=env, stdout=log, stderr=log)
for i in range(60):
    if subprocess.run([OLLAMA_BIN, "list"], env=env, capture_output=True).returncode == 0:
        print(f"✅ server up ({i+1}s)")
        break
    time.sleep(1)

print(f"⏳ pull: {MODEL_NAME}")
t0 = time.time()
subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=env, check=True, timeout=7200)
print(f"✅ pull 完了 ({(time.time()-t0)/60:.1f} 分)")

# manifest から model 層の blob を特定
MANIFEST = os.path.join(
    OLLAMA_MODELS, "manifests/registry.ollama.ai/orcarouter/Qwen3.8-27B-Uncensored/latest")
man = json.load(open(MANIFEST))
MODEL_BLOB = None
for layer in man["layers"]:
    if layer["mediaType"] == "application/vnd.ollama.image.model":
        digest = layer["digest"].replace("sha256:", "sha256-")
        MODEL_BLOB = os.path.join(OLLAMA_MODELS, "blobs", digest)
        print(f"  model 層: {layer['size']/1024**3:.2f} GiB")
        break
if not MODEL_BLOB or not os.path.isfile(MODEL_BLOB):
    raise RuntimeError(f"model blob が見つかりません: {MODEL_BLOB}")
print(f"✅ GGUF: {MODEL_BLOB}")

# Ollama はもう不要（VRAM を空ける）
subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
time.sleep(4)
'''))

cells.append(("markdown", "---\n### 4️⃣ MTP の有無で比較\n"))

cells.append(("code", '''# llama-server を直に起動し、--spec-type を変えて測る
import json, os, re, subprocess, time, urllib.request

LLAMA_SERVER = globals().get("LLAMA_SERVER")
MODEL_BLOB = globals().get("MODEL_BLOB")
PROMPT = "1から20までの素数を列挙して。数字だけをカンマ区切りで。"
CTX = 32768           # f16 KV で 131K を張ると 30 GiB を溢れて CPU に落ちる（実測）
PORT = 9931           # llama.cpp v0.7.0 の新既定
NGPUS = len(VRAM_EACH)
KV_TYPE = "q8_0"      # Ollama 版と同じ。これが無いと VRAM 溢れで 0.8 tps になる


def _free_vram(timeout=180):
    """VRAM が解放されるまで待つ（残留プロセス対策）。"""
    for i in range(timeout):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        used = [int(x) for x in out.splitlines() if x.strip().isdigit()]
        if used and max(used) < 500:
            return i
        time.sleep(1)
    return -1


def _start(spec_type):
    """llama-server を起動して (proc, env) を返す。"""
    subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
    time.sleep(3)
    freed = _free_vram()
    if freed < 0:
        raise RuntimeError("VRAM が解放されませんでした")

    cmd = [
        LLAMA_SERVER,
        "-m", MODEL_BLOB,
        "-c", str(CTX),
        "--cache-type-k", KV_TYPE,
        "--cache-type-v", KV_TYPE,
        "-ngl", "all",
        "-fa", "on",
        "-np", "1",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--jinja",
        "--temp", "0",
    ]
    if spec_type != "none":
        cmd += ["--spec-type", spec_type]

    # Ollama 同梱の llama-server は CUDA バックエンドを同じディレクトリに置く。
    # LD_LIBRARY_PATH を通さないと libggml-cuda.so の dlopen に失敗し、
    # 黙って CPU 推論に落ちる（実測 0.8 tps / VRAM 0 MiB）。
    env = {**os.environ, "LD_LIBRARY_PATH": os.path.dirname(LLAMA_SERVER)}
    log = open(f"/tmp/llama_{spec_type.replace(',','_')}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)

    # 起動待ち
    for i in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return proc, i + 1
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server が落ちました（ctx={CTX}）")
        time.sleep(1)
    raise RuntimeError("llama-server が起動しませんでした")


def _chat(prompt, num_predict=64, timeout=1800):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "n_predict": num_predict,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


results = []

for spec in ["none", "draft-mtp"]:
    print("=" * 64)
    print(f"  --spec-type {spec}")
    print("=" * 64)
    try:
        proc, boot_s = _start(spec)
        print(f"  起動: {boot_s}s")

        # ウォームアップ（計測から除外）
        t0 = time.time()
        _chat("Say OK.", num_predict=8)
        print(f"  ウォームアップ: {time.time()-t0:.1f}s")

        # 本計測
        t0 = time.time()
        d = _chat(PROMPT, num_predict=64)
        wall = time.time() - t0

        usage = d.get("usage") or {}
        timings = d.get("timings") or {}
        ct = usage.get("completion_tokens") or timings.get("predicted_n") or 0
        pt = usage.get("prompt_tokens") or timings.get("prompt_n") or 0
        gen_tps = timings.get("predicted_per_second")
        prm_tps = timings.get("prompt_per_second")
        reply = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")

        row = {
            "spec_type": spec,
            "boot_s": boot_s,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "prompt_tps": round(prm_tps, 1) if prm_tps else None,
            "gen_tps": round(gen_tps, 1) if gen_tps else None,
            "wall_s": round(wall, 1),
            "reply": reply[:80],
        }
        results.append(row)

        print(f"  生成: {ct} tokens / {wall:.1f}s → {row['gen_tps']} tokens/s")
        print(f"  プロンプト処理: {pt} tokens → {row['prompt_tps']} tokens/s")
        print(f"  reply: {row['reply']!r}")

        # GPU オフロードの検証（CPU に落ちていないか）
        log_path = f"/tmp/llama_{spec.replace(',','_')}.log"
        if os.path.isfile(log_path):
            lines = open(log_path, errors="replace").read().splitlines()
            hits = [l for l in lines
                    if any(k in l for k in ("offloaded", "CUDA0", "CUDA1", "buffer size",
                                            "CPU buffer", "no usable", "backend", "ggml_cuda"))]
            if hits:
                for l in hits[:8]:
                    print("  " + l.strip()[:150])
            else:
                print("  ⚠ バックエンドの記載なし。ログ先頭:")
                for l in lines[:6]:
                    print("    " + l.strip()[:140])

        # 推論中の VRAM 使用量
        vram = " / ".join(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.split())
        print(f"  VRAM 使用: {vram} MiB")

        subprocess.run(["pkill", "-9", "llama-server"], capture_output=True)
        time.sleep(3)

    except Exception as e:
        print(f"  ❌ 失敗: {e}")
        log_path = f"/tmp/llama_{spec.replace(',','_')}.log"
        if os.path.isfile(log_path):
            print("  --- ログ末尾 ---")
            print(open(log_path).read()[-2000:])
        results.append({"spec_type": spec, "error": str(e)})

print()
print("=" * 64)
print("  結果まとめ")
print("=" * 64)
print(f"{'spec-type':>12} | {'boot':>6} | {'prompt tps':>10} | {'gen tps':>8} | {'wall':>7}")
print("-" * 60)
for r in results:
    if "error" in r:
        print(f"{r['spec_type']:>12} | {'—':>6} | {'—':>10} | {'—':>8} | ERROR")
    else:
        print(f"{r['spec_type']:>12} | {r['boot_s']:>5}s | {r['prompt_tps']:>10} | "
              f"{r['gen_tps']:>8} | {r['wall_s']:>6.1f}s")

base = next((r for r in results if r.get("spec_type") == "none" and "gen_tps" in r), None)
mtp = next((r for r in results if r.get("spec_type") == "draft-mtp" and "gen_tps" in r), None)
if base and mtp and base["gen_tps"] and mtp["gen_tps"]:
    print()
    print(f"  MTP の効果: {base['gen_tps']} → {mtp['gen_tps']} tokens/s "
          f"（{mtp['gen_tps']/base['gen_tps']:.2f}x）")

print()
print("JSON:")
print(json.dumps(results, ensure_ascii=False, indent=2))
'''))

cells.append(("markdown", """---
### 5️⃣ 読み方

- **gen tps** が実用速度。MTP でどこまで伸びるかを見ます
- MTP は投機的デコードなので、**当たれば速く、外れればほぼ同じ**です
- プロンプト処理（prompt tps）は MTP の影響を受けません
- `--spec-type draft-mtp` が無いバイナリでは、この検証は成立しません（セル2で判定）

### 注意
- Ollama 版（13.7 tokens/s @ 131K）との比較は**条件が違う**ので単純比較できません
  - Ollama は `OLLAMA_KV_CACHE_TYPE=q8_0`、ここでも `--cache-type-k/v q8_0` に合わせる
  - **f16 KV で 131K を張ると 30 GiB を溢れて CPU 推論に落ちる**（実測 0.8 tps）。
    そのため ctx は 32768 で固定し、`--spec-type` の差だけを見る
  - **`LD_LIBRARY_PATH` を llama-server のディレクトリに通すこと。**
    これが無いと CUDA バックエンドの dlopen に失敗し、VRAM 0 MiB のまま
    CPU 推論に落ちる（エラーは出ないので気づきにくい）
  - サンプリング設定も異なる
- 純粋な MTP の効果は **同一バイナリ内の none ↔ draft-mtp の差**で見てください
"""))

pack_nb.write(OUT, cells, cell_ids=True)
