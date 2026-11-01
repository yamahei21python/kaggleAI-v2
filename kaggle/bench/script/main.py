#!/usr/bin/env python3
"""bench ctx vs speed orcarouter qwen3 8 27b script（自動生成。直接編集しない）。

生成: kaggle/build_batch_script.py ← kaggle/build_bench.py のセル列
"""


# ── 📊 Benchmark: コンテキスト長 vs 速度 ──────────────────


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


def kc_vram_used(timeout=5):
    """GPU ごとの使用 VRAM (MiB)。取れなければ []。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def kc_sample(pid, root="/proc"):
    """1 プロセスの様子を 1 つの dict にまとめる。取れない項目は None。

    `read_mib` は **実際にブロックデバイスまで行った読み込み量**（/proc の
    read_bytes）。ページキャッシュに当たった分は含まれないので、
    「重みを読み終えたか」はこちらで見る。`char_mib` は read(2) の総量で、
    キャッシュ命中も含む。両方出すのは、この 2 つが離れたときに
    「キャッシュから読んでいる」と言い切れるようにするため。

    `root` はテスト用の口。実プロセスの RSS は環境で変わるので、そこを
    assert すると「たまたま通る」テストになる。`/proc` の形をした
    ディレクトリを渡せば、変換（kB → MiB など）だけを固定できる。
    """
    s = {"rss_mib": None, "read_mib": None, "char_mib": None, "vram_mib": None}
    try:
        with open(f"{root}/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    s["rss_mib"] = round(int(line.split()[1]) / 1024)
                    break
    except OSError:
        pass
    try:
        with open(f"{root}/{pid}/io") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    s["read_mib"] = int(line.split()[1]) // 2**20
                elif line.startswith("rchar:"):
                    s["char_mib"] = int(line.split()[1]) // 2**20
    except OSError:
        pass
    vram = kc_vram_used()
    if vram:
        s["vram_mib"] = sum(vram)
    return s


def kc_watch(pid, every=30, keep=80):
    """待機中のプロセスを `every` 秒ごとに観測する。戻り値は `(samples, stop)`。

    `stop()` を呼ぶと観測を止める。**プロセスの死だけでは止まらない**設計に
    してあるのは、待っている相手がサーバー本体だから: 起動が通れば同じ
    PID が何時間も生き続けるので、死を待つと `[watch]` が 30 秒ごとに
    何百行も出てログを埋める。呼び出し側は待機が終わった時点で止めること。

    無言の待機を「遅い」と「固まった」に分けるための計器。実測 2026-11-01
    exl3: 1800 秒の待機中に chat_server.log が **0 バイト**で、読み込みが
    進んでいたのか止まっていたのかまったく分からなかった（エンジン側の
    進捗表示は `--no-progress` で切ってあり、完走時の 1 行しか出ない）。
    そうなると次の手は「もう 1 回回す」しかなく、1 往復 30 分を捨てる。

    観測は read_bytes の**増分**が主役。1800 秒で 3 GiB しか進んでいなければ
    1.7 MiB/s ということなので、遅いのか止まったのかは即座に判る。

    戻り値の samples はそのまま result.json に入れる。失敗時にも
    「どこまで進んでいたか」が残るのが目的なので、成功・失敗を問わず入れる。
    """
    import os
    import threading

    samples = []
    stop = threading.Event()

    def loop():
        t0 = time.time()
        while not stop.wait(every):        # 止める指示が来たら即座に抜ける
            s = kc_sample(pid)
            s["t"] = round(time.time() - t0)
            samples.append(s)
            del samples[:-keep]
            print(f"  [watch] {s['t']}s rss={s['rss_mib']}MiB "
                  f"read={s['read_mib']}MiB vram={s['vram_mib']}MiB",
                  flush=True)
            try:
                os.kill(pid, 0)      # 死んでいれば ProcessLookupError
            except OSError:
                return

    threading.Thread(target=loop, daemon=True).start()
    return samples, stop.set


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
    """"-x cloudflared" → ["-x", "cloudflared"]（shell を介さない）。"""
    return pattern.split()


def kc_pids(pattern, timeout=15):
    """pgrep の引数文字列（例: "-x cloudflared"）に一致する PID を返す。"""
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


GPU_NAMES, VRAM_EACH = kc_detect_gpu()
VRAM_TOTAL = sum(VRAM_EACH)
print(f"GPU  : {GPU_NAMES}")
print(f"VRAM : {VRAM_EACH} MiB each → 合計 {VRAM_TOTAL} MiB ({VRAM_TOTAL/1024:.1f} GiB)")
if VRAM_TOTAL == 0:
    raise RuntimeError("GPU なし。Accelerator を 'GPU T4 x2' に設定してください。")


# ── 2️⃣ Ollama インストール + モデル pull ────────────────


import os, subprocess, time

MODEL_NAME = "orcarouter/Qwen3.8-27B-Uncensored:latest"
OLLAMA_MODELS = "/kaggle/tmp/ollama_models"
os.makedirs(OLLAMA_MODELS, exist_ok=True)

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd",
               shell=True, check=False)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
               shell=True, check=True, timeout=900)
print("✅ Ollama installed")

OLLAMA_BIN = "/usr/local/bin/ollama"

# pull は 1 回だけでよい（コンテキストは起動時に決まる）
base_env = {**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS}
with open("/tmp/ollama_boot.log", "w") as log:
    subprocess.Popen([OLLAMA_BIN, "serve"], env=base_env, stdout=log, stderr=log)
for i in range(60):
    if subprocess.run([OLLAMA_BIN, "list"], env=base_env,
                      capture_output=True).returncode == 0:
        print(f"✅ server up ({i+1}s)")
        break
    time.sleep(1)

print(f"⏳ pull: {MODEL_NAME}")
t0 = time.time()
subprocess.run([OLLAMA_BIN, "pull", MODEL_NAME], env=base_env,
               check=True, timeout=7200)
print(f"✅ pull 完了 ({(time.time()-t0)/60:.1f} 分)")


# ── 3️⃣ ベンチマーク本体 ────────────────────────────────


# 各コンテキスト長で Ollama を再起動 → 同じプロンプトで計測
import os, json, subprocess, time

OLLAMA_BIN = "/usr/local/bin/ollama"
PROMPT = "1から20までの素数を列挙して。数字だけをカンマ区切りで。"
CTX_LIST = [32768, 131072, 262144]


def _free_vram(timeout=180):
    """llama-server が VRAM を手放すまで待つ。戻り値は待った秒数（失敗は -1）。"""
    return kc_wait_gpu_idle(timeout=timeout)


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
    return kc_ollama_chat(11434, MODEL_NAME, prompt,
                          num_predict=num_predict, timeout=timeout, think=False)


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


# ── 4️⃣ 読み方 ─────────────────────────────────────
