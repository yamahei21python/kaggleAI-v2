# モデルを VRAM にロードして疎通確認
# 初回は 2〜5 分かかります（16.5 GiB をロード）
import json, time, urllib.request

def _post(path, payload, timeout=900):
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

print(f"⏳ ウォームアップ中（コンテキスト {CONTEXT_LENGTH:,}）...")
t0 = time.time()
try:
    resp = _post("/api/chat", {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": False,
        # 16 だと thinking（推論トークン）だけで予算を使い切り、
        # content が空のまま返る（実測 reply=''）。生成証明には十分な余量を。
        "options": {"num_predict": 128, "temperature": 0},
    })
    dt = time.time() - t0
    print(f"✅ 応答あり（{dt:.1f}s）")
    print(f"   load : {(resp.get('load_duration') or 0)/1e9:.1f}s")
    print(f"   reply: {resp.get('message', {}).get('content', '')!r}")
    ec = resp.get("eval_count") or 0
    ed = (resp.get("eval_duration") or 1) / 1e9
    print(f"   速度 : {ec/ed:.1f} tokens/s" if ec else "   速度 : N/A")
    if not (resp.get("message", {}).get("content")
            or resp.get("message", {}).get("thinking")):
        # HTTP 成功だが生成物が空 = verify [2] と同じ症状の予兆なので可視化する
        print("   ⚠ content/thinking が空（verify [2] で判定されます）")
    # GPU 常駐検査。CPU フォールバックでも HTTP 200 は返るので、
    # 通信だけ見ていると 3/3 PASS のまま静かに劣化する。
    import subprocess as _sp
    _ps = _sp.run([OLLAMA_BIN, "ps"], env=env,
                  capture_output=True, text=True).stdout.strip().splitlines()
    print("   ps :", " | ".join(l.strip() for l in _ps[1:] if l.strip()) or "(空)")
    _psrow = " ".join(l for l in _ps[1:] if "%" in l)
    if _psrow and "GPU" not in _psrow:
        print("   ⚠ プロセッサ表示に GPU が無い → CPU 実行の可能性")
    # ollama ps の CONTEXT 列はカンマ無し（131072）。正規化して比較する。
    if _psrow and str(CONTEXT_LENGTH) not in _psrow.replace(",", ""):
        print(f"   ⚠ コンテキストが {CONTEXT_LENGTH:,} と異なる（ps 表示: {_psrow}）")
except Exception as e:
    print(f"⚠ ウォームアップ失敗: {e}（LOAD_TIMEOUT={env.get('OLLAMA_LOAD_TIMEOUT')}s）")
    print(open("/tmp/ollama.log").read()[-3000:])
    # 握り潰すと verify が「原因不明の FAIL」になるだけなので、ここで止める
    fail(f"ウォームアップ失敗: {type(e).__name__}: {e}", "/tmp/ollama.log")
