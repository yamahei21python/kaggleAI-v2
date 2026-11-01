# vLLM 固有の確認項目（足場は kaggle/serve-cells/verify.py が持つ）
# 呼び出し側: engine_checks(judge, retry, base)
import json, subprocess as _sp, time as _time
import urllib.request as _urlreq

# 骨組みの verify() が auth を渡す。notebook 版は認証 OFF なので None のまま。
AUTH = None


def _req(url, timeout):
    """AUTH があれば Authorization を付けた Request を返す。"""
    if not AUTH:
        return url
    return _urlreq.Request(url, headers={"Authorization": AUTH})


def _post(url, payload, timeout=600):
    # ngrok のブラウザ警告ページを踏まないヘッダは kc_post_json が付ける。
    # IPv4 フォールバックも効かせる（Kaggle VM は AAAA だけ返すことがある）。
    return kc_post_json(url, payload, timeout=timeout, opener=kc_urlopen_v4,
                        headers={"Authorization": AUTH} if AUTH else None)


CHAT_BODY = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
}


def _reply_parts(j):
    """content と reasoning_content を分けて返す（thinking モデル対策）。"""
    m = (j.get("choices") or [{}])[0].get("message") or {}
    return (m.get("content") or "").strip(), (m.get("reasoning_content") or "").strip()


def engine_checks(judge, retry, base, auth=None, direct=False):
    global AUTH
    AUTH = auth
    # [1] 認証なしでチャット UI が取れる（/ は 200 で HTML）
    #     UI は proxy が返すもの。smoke（direct=True）は proxy を立てないので
    #     `/` はエンジン自身の応答（vLLM は 404）になる。誤判定を避けて SKIP する。
    print("[1] 認証なしで UI")
    if direct:
        judge(1, "ui", None, "skip: エンジン直（proxy 無し。smoke）")
    else:
        try:
            with retry(lambda: kc_urlopen_v4(f"{base}/", timeout=60)) as r:
                html = r.read()
                # UI 判定は HTML であること + 十分なサイズで行う（文言に依存させない）
                ok_ui = (r.status == 200 and len(html) > 10000
                         and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
                judge(1, "ui", ok_ui, f"HTTP {r.status} {len(html)}B")
        except Exception as e:
            judge(1, "ui", False, f"{type(e).__name__}: {e}")

    # [2] /v1/models（vLLM が応答している証明）
    print("[2] /v1/models")
    try:
        with retry(lambda: kc_urlopen_v4(_req(f"{base}/v1/models", 60), timeout=60)) as r:
            md = json.loads(r.read())
        ids = [m.get("id") for m in md.get("data", [])]
        judge(2, "models", r.status == 200 and MODEL_NAME in ids,
              f"HTTP {r.status} ids={ids}")
    except Exception as e:
        judge(2, "models", False, f"{type(e).__name__}: {e}")

    # [3] ストリーミング（SSE がトンネルを抜けるか）
    print("[3] ストリーミング")
    try:
        resp = retry(lambda: _post(f"{base}/v1/chat/completions",
                                   dict(CHAT_BODY, stream=True)))
        times = []
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
        judge(3, "streaming", ok, detail)
    except Exception as e:
        judge(3, "streaming", False, f"{type(e).__name__}: {e}")

    # [4] OpenAI 互換（temperature=0 を送っても 200 で生成される）
    #     プロキシの --enforce-sampling が 0.7 に矯正する経路の確認でもある。
    print("[4] /v1/chat/completions (temperature=0)")
    try:
        body4 = dict(CHAT_BODY, temperature=0, repetition_penalty=1.0)
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 body4, timeout=300)) as r:
            st4 = r.status
            cj = json.loads(r.read())
        ctext = "".join((c.get("message") or {}).get("content") or ""
                        for c in cj.get("choices", []))
        usage = cj.get("usage") or {}
        ok4 = st4 == 200 and bool(ctext or usage.get("completion_tokens"))
        judge(4, "openai-compat", ok4,
              f"HTTP {st4} reply={ctext[:40]!r} usage={usage}")
    except Exception as e:
        judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

    # [5] GPU 常駐（CPU フォールバック検出）
    print("[5] GPU 常駐")
    try:
        mem = _sp.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        used = [int(x.split(",")[0]) for x in mem.splitlines() if "," in x]
        alive = _sp.run(["pgrep", "-f", "vllm serve"],
                        capture_output=True, text=True).stdout.strip() != ""
        judge(5, "gpu-resident", alive and bool(used) and sum(used) > 4000,
              f"proc={'alive' if alive else 'DEAD'} vram={used}")
    except Exception as e:
        judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

    # [6] 会話（マルチターン: 履歴を渡して文脈が保たれるか）
    print("[6] 会話（マルチターン）")
    try:
        conv = [{"role": "user",
                 "content": "私の名前はコウヘイです。覚えてください。短く返事して。"}]
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 dict(CHAT_BODY, messages=conv,
                                      max_tokens=96), timeout=300)) as r1:
            j1 = json.loads(r1.read())
        c1, t1 = _reply_parts(j1)
        # thinking だけで content が空のことがある。履歴には本文を優先し、
        # 無ければ思考の先頭を使う（空の assistant を積むと文脈が切れる）。
        conv.append({"role": "assistant", "content": c1 or t1[:600] or "(empty)"})
        conv.append({"role": "user", "content": "私の名前は何？名前だけ答えて。"})
        with retry(lambda: _post(f"{base}/v1/chat/completions",
                                 dict(CHAT_BODY, messages=conv,
                                      max_tokens=96), timeout=300)) as r2:
            j2 = json.loads(r2.read())
        c2, t2 = _reply_parts(j2)
        a2 = (c2 + "\n" + t2).strip()
        name_ok = any(x in a2.lower() for x in ("コウヘイ", "コーヘイ", "kohei"))
        judge(6, "conversation", bool(c1 or t1) and name_ok,
              f"turn1={(c1 or t1)[:50]!r} turn2={a2[:70]!r}")
    except Exception as e:
        judge(6, "conversation", False, f"{type(e).__name__}: {e}")
