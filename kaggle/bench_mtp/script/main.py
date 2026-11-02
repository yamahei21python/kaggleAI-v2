#!/usr/bin/env python3
"""bench mtp orcarouter qwen3 8 27b script（自動生成。直接編集しない）。

生成: kaggle/build_batch_script.py ← kaggle/build_bench_mtp.py のセル列
"""


# ── 📊 Benchmark: llama-server 直起動 + MTP 投機的デコード ─


# ── 🧩 共通ランタイム（GPU / HTTP） ───────────────────────


# ============================================================
# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）
# proxy / tunnel / keepalive / verify の配管をここに集約する。
# ============================================================
# ── runtime/net.py ────────────────────────────────────────
"""トンネル URL 取得と HTTP 疎通の共通処理（生成コードへ埋め込む）。"""
import json
import re
import socket
import time
import urllib.error
import urllib.request

# hosts は表駆動。今はログ解析の 1 種類しか無いが、待ち時間を URL の出所と
# 一緒に置いておくと `kc_wait_url` が既定値を知らずに済む。表に無い名前は
# KeyError になる（黙って空 URL を返さない）。
TUNNEL_HOSTS = {
    "cloudflared": {
        "url_re": re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com"),
        "wait_seconds": 90,
    },
}


# Dataset 同梱バイナリと同じ版に固定する（latest は再現性がない）。
KC_CLOUDFLARED_VERSION = "2026.10.4"


def kc_cloudflared_url(version=KC_CLOUDFLARED_VERSION):
    """cloudflared (linux amd64) のダウンロード URL。"""
    return ("https://github.com/cloudflare/cloudflared/releases/download/"
            f"{version}/cloudflared-linux-amd64")


def kc_probe_url(host, log_path=None):
    """現在の公開URLを返す（無ければ ""）。host は cloudflared のみ。"""
    spec = TUNNEL_HOSTS[host]
    if not log_path:
        return ""
    try:
        with open(log_path) as f:
            m = spec["url_re"].search(f.read())
        return m.group(0) if m else ""
    except OSError:
        return ""


def kc_wait_url(host, log_path=None, seconds=None, interval=1, progress_every=10):
    """公開URLが出るまで待つ。seconds 未指定なら host ごとの既定値。"""
    if seconds is None:
        seconds = TUNNEL_HOSTS[host]["wait_seconds"]
    for i in range(seconds):
        url = kc_probe_url(host, log_path)
        if url:
            return url
        if progress_every and i % progress_every == progress_every - 1:
            print(f"⏳ URL 待ち ({i + 1}/{seconds})...")
        time.sleep(interval)
    return ""


def kc_urlopen_v4(url, timeout=15):
    """公開 URL を開く。失敗したら IPv4 優先で名前解決して 1 回だけ再試行。

    Kaggle の VM は AAAA（IPv6）を返すのに IPv6 経路が無いことがあり、
    `urlopen` が "Network is unreachable" で落ちる（dryrun で実測）。
    自分の URL に届かないだけでトンネルは生きているので、ここで諦めると
    keepalive が**生きているトンネルを再起動して URL が変わってしまう**。
    """
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


def kc_wait_http(url, attempts=20, interval=3, timeout=10, opener=None):
    """200 が返るまで待つ。成功したら経過秒（(i+1)*interval）、失敗なら None。"""
    opener = opener or urllib.request.urlopen
    for i in range(attempts):
        try:
            with opener(url, timeout=timeout) as r:
                if r.status == 200:
                    return (i + 1) * interval
        except Exception:
            pass
        time.sleep(interval)
    return None


def kc_retry(fn, tries=6, delay=5, skip_http_errors=True):
    """一過性の通信エラーを吸収する。HTTP 応答は「答え」なので既定で再試行しない。"""
    last = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if skip_http_errors:
                raise
            last = e
        except Exception as e:
            last = e
        if i + 1 < tries:
            wait = delay * (i + 1)
            print(f"  ⚠ 一時的な通信エラー（{i + 1}/{tries}）: {last} "
                  f"→ {wait}s 待って再試行")
            time.sleep(wait)
    raise last


def kc_post_json(url, payload, timeout=60, headers=None, opener=None):
    """JSON POST のレスポンスを返す（ストリームは呼び出し側で読む）。"""
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=h, method="POST")
    return (opener or urllib.request.urlopen)(req, timeout=timeout)

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


# ── 1️⃣ GPU 確認 ──────────────────────────────────


GPU_NAMES, VRAM_EACH = kc_detect_gpu()
VRAM_TOTAL = sum(VRAM_EACH)
print(f"GPU  : {GPU_NAMES}")
print(f"VRAM : {VRAM_EACH} MiB each → 合計 {VRAM_TOTAL} MiB ({VRAM_TOTAL/1024:.1f} GiB)")
if VRAM_TOTAL == 0:
    raise RuntimeError("GPU なし。Accelerator を 'GPU T4 x2' に設定してください。")


# ── 2️⃣ llama-server バイナリの入手 ────────────────────


# llama-server を単体で入手する。
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


# ── 3️⃣ モデル GGUF を入手 ────────────────────────────


# Ollama で pull してから blob から GGUF を取り出す。
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


# ── 4️⃣ MTP の有無で比較 ──────────────────────────────


# llama-server を直に起動し、--spec-type を変えて測る
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
    return kc_wait_gpu_idle(timeout=timeout)


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
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0,
        "n_predict": num_predict,
        "cache_prompt": False,
    }
    resp = kc_post_json(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                        timeout=timeout)
    with resp:
        return json.loads(resp.read())


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


# ── 5️⃣ 読み方 ─────────────────────────────────────
