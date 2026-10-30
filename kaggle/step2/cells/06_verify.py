# 公開URL経由の疎通確認（UI / 200 / streaming / OpenAI互換 / GPU常駐の5点）
# 結果は /kaggle/working/verify_result.txt に残す（save_output が /output にも複製）。
# 失敗は raise（検証不能のまま放置しない）。
import json, time as _time, urllib.request, urllib.error

try:
    with open("/tmp/public_url.txt") as f:
        public_url = f.read().strip()
except FileNotFoundError:
    fail("公開URLがありません。トンネルセルを実行してください。")

# ---- トンネル確立の取り直し ----
# tunnel セルでエッジ伝播待ちが終わっていなくても、ここで吸収する。
# 届かなければ [1] がそのまま FAIL になる（握り潰さない）。
_elapsed = kc_wait_http(f"{public_url}/healthz", attempts=10, interval=3)
if _elapsed is not None:
    if _elapsed > 3:
        print(f"✅ トンネル確立をここで確認（{_elapsed}s）")
else:
    print("⚠ /healthz に届きません。トンネル未確立の可能性（[1] で判定）")
lines = [
    f"PUBLIC_URL={public_url}",
    f"MODEL={MODEL_NAME}",
    f"TIME={_time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]
failures = []


def _call(url, timeout=900, stream=False):
    # ngrok 無料枠のブラウザ向け警告ページを踏まないためのヘッダ（cloudflared では無害）
    headers = {"Content-Type": "application/json",
               "ngrok-skip-browser-warning": "true"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Say OK."}],
        "stream": stream,
        # 非ストリームの 16 は thinking だけで枯渇し content が空になる
        # （実測 reply=''）。十分な余量を与える。
        "options": {"num_predict": 128 if not stream else 64, "temperature": 0},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _judge(idx, label, ok, detail):
    mark = "✅" if ok else "❌"
    print(f"{mark} {label}: {detail}")
    lines.append(f"[{idx}] {label} -> {'PASS' if ok else 'FAIL'}: {detail}")
    if not ok:
        failures.append(f"[{idx}] {label}: {detail}")


def _with_retry(fn, tries=6, delay=5):
    # 実行時ログで確認した瞬間的障害: URLError [Errno 101] Network is
    # unreachable / [-2] Name or service not known（v11 で verify [2][3] が
    # 約1分の DNS/経路瞬断で落ち、[1][4] は通過＝経路は生きていた）。
    # 3回/4秒（~8秒）では吸収できないので、待ちを 5,10,15,20,25 秒と
    # 伸ばして合計 75 秒の窓を持つ。HTTP 応答（401/500 等）は「答え」
    # なので再試行せず、そのまま判定に渡す。
    return kc_retry(fn, tries=tries, delay=delay)


# [1] 認証なしでチャット UI が取れる（/ は 200 で HTML）
print("[1] 認証なしで UI")
try:
    with _with_retry(lambda: urllib.request.urlopen(
            f"{public_url}/", timeout=60)) as r:
        html = r.read()
        # UI 判定は HTML であること + 十分なサイズで行う。
        # 旧実装は b"Kaggle Chat" を見ていたが、HTML 側のタイトル変更で
        # 常に FAIL になる（実測）。文言に依存させない。
        ok1 = (r.status == 200 and len(html) > 10000
               and html.lstrip()[:15].lower().startswith(b"<!doctype html"))
        _judge(1, "ui", ok1, f"HTTP {r.status} {len(html)}B")
except Exception as e:
    _judge(1, "ui", False, f"{type(e).__name__}: {e}")

# [2] 認証なしで /api/chat が 200 + 生成
print("[2] /api/chat")
try:
    with _with_retry(lambda: _call(f"{public_url}/api/chat")) as r:
        resp = json.loads(r.read())
        reply = (resp.get("message") or {}).get("content", "")
        think = (resp.get("message") or {}).get("thinking", "")
        # 生成証明 = content か thinking か評価トークンのどれかが実在する
        #（thinking モデルは content が空でも生成は成功している）。
        made = bool(reply or think or (resp.get("eval_count") or 0))
        _judge(2, "chat-ok", r.status == 200 and made,
               f"HTTP {r.status} reply={reply!r} "
               f"eval={resp.get('eval_count')}")
except Exception as e:
    _judge(2, "auth-ok", False, f"{type(e).__name__}: {e}")

# [3] ストリーミング（chunked が cloudflared を通り抜けるか）
#     ここが通らなければ token 逐次出力 = 通常の利用形態が壊れる。
print("[3] ストリーミング")
try:
    times = []
    # 接続確立を先に終えてから計測を始める（再試行の sleep が first/total に
    # 混ざると逐次判定が偽 FAIL になる）
    resp3 = _with_retry(lambda: _call(f"{public_url}/api/chat",
                                      timeout=600, stream=True))
    t0 = _time.time()
    with resp3 as r:
        for raw in r:            # 改行が届くまで待つ = 逐次受信になっている証拠
            if not raw.strip():
                continue
            times.append(_time.time() - t0)
    total = _time.time() - t0
    if len(times) >= 3:
        first = times[0]
        spread = times[-1] - first
        # 到着分布で判定（生成長に依存しない）:
        #  - 全バッファ転送なら first ≈ total（first < total*0.8 が偽）。
        #  - 3 到着以上 & 最初と最後の間隔が 0.2s 超 = 逐次の到着証拠。
        # span>0.5 の旧基準は短い生成で偽 FAIL になった。
        ok = spread > 0.2 and first < total * 0.8
        detail = (f"chunks={len(times)} first={first:.2f}s "
                  f"total={total:.2f}s spread={spread:.2f}s")
    else:
        ok, detail = False, f"chunks={len(times)} first=N/A"
    _judge(3, "streaming", ok, detail)
except Exception as e:
    _judge(3, "streaming", False, f"{type(e).__name__}: {e}")

# [4] OpenAI 互換経路（配布する opencode.json が使うのはこちら）
print("[4] /v1/chat/completions")
try:
    req4 = urllib.request.Request(
        f"{public_url}/v1/chat/completions",
        data=json.dumps({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 128,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "ngrok-skip-browser-warning": "true"},
        method="POST",
    )
    with _with_retry(lambda: urllib.request.urlopen(req4, timeout=300)) as r:
        st4 = r.status
        cj = json.loads(r.read())
    ctext = "".join((c.get("message") or {}).get("content") or ""
                    for c in cj.get("choices", []))
    usage = cj.get("usage") or {}
    ok4 = st4 == 200 and bool(ctext or usage.get("completion_tokens"))
    _judge(4, "openai-compat", ok4,
           f"HTTP {st4} reply={ctext[:40]!r} usage={usage}")
except Exception as e:
    _judge(4, "openai-compat", False, f"{type(e).__name__}: {e}")

# [5] GPU 常駐 + コンテキスト長（CPU フォールバックの検出）
#     通信だけ見ていると 3/3 PASS のまま VRAM に載っていないことがある。
print("[5] GPU 常駐")
try:
    import subprocess as _sp
    ps_out = _sp.run(["/usr/local/bin/ollama", "ps"],
                     env=env, capture_output=True, text=True, timeout=30).stdout
    # データ行だけに絞る（ヘッダは "GPU" 文字を含まない）。
    # ollama ps の CONTEXT 列はカンマ無し（例: 131072）なので両方正規化して比較。
    rows = [l.strip() for l in ps_out.splitlines()[1:] if "%" in l]
    row = " | ".join(rows)
    # 完全 CPU フォールバックは "100% CPU" で GPU 文字が無い。
    # 分割オフロード（x% GPU/y% CPU）は VRAM 常駐の一部として許容する。
    has_gpu = "GPU" in row
    has_ctx = str(CONTEXT_LENGTH) in row.replace(",", "")
    _judge(5, "gpu-resident", bool(row) and has_gpu and has_ctx,
           f"ps={row[:160]!r} ctx_ok={has_ctx}")
except Exception as e:
    _judge(5, "gpu-resident", False, f"{type(e).__name__}: {e}")

# ---- レポート ----
report = "\n".join(lines)
print()
print("=" * 60)
print(report)
print("=" * 60)
if not save_output("verify_result.txt", report + "\n"):
    # 保存失敗 = 証跡なし。ログは取れないので成功扱いにしてはいけない。
    failures.append("verify_result.txt を永続領域に保存できなかった")

if failures:
    fail("疎通確認に失敗: " + "; ".join(failures), TUNNEL_LOG)
print("✅ 5/5 パス — UI / 200 / streaming / OpenAI互換 / GPU常駐")
