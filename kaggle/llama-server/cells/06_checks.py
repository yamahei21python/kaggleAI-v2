# llama-server 直起動の確認項目（足場は kaggle/serve-cells/verify.py が持つ）
# 呼び出し側: engine_checks(judge, retry, base)
import json, subprocess as _sp, time as _time, urllib.request

# 骨組みの verify() が auth を渡す。notebook 版は認証 OFF なので None のまま。
AUTH = None


def _auth_headers():
    """AUTH があれば Authorization を足したヘッダを返す。"""
    h = {"Content-Type": "application/json"}
    if AUTH:
        h["Authorization"] = AUTH
    return h


def _post(url, payload, timeout=900):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers=_auth_headers(),
        method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


CHAT_BODY = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0,
}


def engine_checks(judge, retry, base, auth=None, direct=False):
    global AUTH
    AUTH = auth
    # [1] 認証なしでチャット UI が取れる（/ は 200 で HTML）
    #     UI は proxy が返すもの。smoke（direct=True）は proxy を立てないので
    #     `/` はエンジン自身の応答になる。誤判定を避けて SKIP する。
    print("[1] 認証なしで UI")
    if direct:
        judge(1, "ui", None, "skip: エンジン直（proxy 無し。smoke）")
    else:
        try:
            with retry(lambda: urllib.request.urlopen(f"{base}/", timeout=60)) as r:
                html = r.read()
                # UI 判定は HTML であること + 十分なサイズで行う（文言に依存させない）
                ok1 = (r.status == 200 and len(html) > 10000
                       and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
                judge(1, "ui", ok1, f"HTTP {r.status} {len(html)}B")
        except Exception as e:
            judge(1, "ui", False, f"{type(e).__name__}: {e}")

    # [2] /v1/models（llama-server が応答している証明）
    print("[2] /v1/models")
    try:
        with retry(lambda: urllib.request.urlopen(
                f"{base}/v1/models", timeout=60)) as r:
            md = json.loads(r.read())
        ids = [m.get("id") for m in md.get("data", [])]
        judge(2, "models", r.status == 200 and MODEL_NAME in ids,
              f"HTTP {r.status} ids={ids}")
    except Exception as e:
        judge(2, "models", False, f"{type(e).__name__}: {e}")

    # [3] 短文応答（生成証明 = content / reasoning / usage のどれかが実在）
    print("[3] /v1/chat/completions")
    try:
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 dict(CHAT_BODY, max_tokens=16),
                                 timeout=600)) as r:
            cj = json.loads(r.read())
        msg = (cj.get("choices") or [{}])[0].get("message") or {}
        ctext = (msg.get("content") or "").strip()
        think = (msg.get("reasoning_content") or "").strip()
        usage = cj.get("usage") or {}
        made = bool(ctext or think or usage.get("completion_tokens"))
        judge(3, "chat-ok", r.status == 200 and made,
              f"HTTP {r.status} reply={ctext[:40]!r} usage={usage}")
    except Exception as e:
        judge(3, "chat-ok", False, f"{type(e).__name__}: {e}")

    # [4] ストリーミング（chunked がトンネルを通り抜けるか）
    print("[4] ストリーミング")
    try:
        times = []
        # 接続確立を先に終えてから計測を始める（再試行の sleep を混ぜない）
        resp = retry(lambda: _post(f"{base}/v1/chat/completions",
                                   dict(CHAT_BODY, stream=True),
                                   timeout=600))
        t0 = _time.time()
        with resp as r:
            for raw in r:            # 行到着 = 逐次受信の証拠
                if not raw.strip():
                    continue
                times.append(_time.time() - t0)
        total = _time.time() - t0
        if len(times) >= 3:
            first = times[0]
            spread = times[-1] - first
            ok = spread > 0.2 and first < total * 0.8
            detail = (f"chunks={len(times)} first={first:.2f}s "
                      f"total={total:.2f}s spread={spread:.2f}s")
        else:
            ok, detail = False, f"chunks={len(times)} first=N/A"
        judge(4, "streaming", ok, detail)
    except Exception as e:
        judge(4, "streaming", False, f"{type(e).__name__}: {e}")

    # [5] GPU 常駐（CPU フォールバックでも HTTP 200 は返るので必ず見る）
    print("[5] GPU 常駐")
    try:
        mem = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used = [int(x.split(",")[0]) for x in mem.splitlines() if "," in x]
        alive = _sp.run(["pgrep", "-f", "llama-server"],
                        capture_output=True, text=True).stdout.strip() != ""
        judge(5, "gpu-resident", alive and bool(used) and sum(used) > 4000,
              f"proc={'alive' if alive else 'DEAD'} vram={used}")
    except Exception as e:
        judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

    # [6] 実効速度（MTP の効果測定。128 tokens を 2 回、2 回目が実効値）
    print("[6] 速度測定（128 tokens 生成）")
    try:
        rates = []
        for run in range(2):
            t0 = _time.time()
            with retry(lambda: _post(f"{base}/v1/chat/completions", {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user",
                                  "content": "Write a short paragraph about the ocean."}],
                    "max_tokens": 128,
                    "temperature": 0,
            }, timeout=900)) as r:
                cj = json.loads(r.read())
            dt = _time.time() - t0
            ct = (cj.get("usage") or {}).get("completion_tokens") or 0
            rates.append(ct / dt if dt > 0 else 0.0)
            print(f"    run{run+1}: {ct} tokens / {dt:.2f}s = {rates[-1]:.1f} t/s")
        # MTP の有無で速度が変わるかを後から比較できるよう成果物に残す
        save_output("bench_direct.json", json.dumps({
            "mtp": USE_MTP,
            "context": CONTEXT_LENGTH,
            "rates": rates,
            "effective_tps": rates[-1],
        }, indent=2) + "\n")
        judge(6, "speed", rates[-1] > 1.0,
              f"effective={rates[-1]:.1f} t/s (MTP={'on' if USE_MTP else 'off'})")
    except Exception as e:
        judge(6, "speed", False, f"{type(e).__name__}: {e}")
