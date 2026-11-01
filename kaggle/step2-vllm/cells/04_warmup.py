# ダミー生成で疎通確認 + 速度計測
import json
import subprocess
import time
import urllib.request

payload = {
    "model": SERVED_NAME,
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 64,
    "temperature": 0.7,
    "repetition_penalty": 1.05,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
print("⏳ ウォームアップ中...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
except Exception as e:
    # 起動ログは試行番号で名前が変わる。読めなくても失敗自体は伝える。
    for _i in range(1, 7):
        _p = "/tmp/vllm_%d.log" % _i
        try:
            print(f"--- {_p} ---")
            print(open(_p, errors="replace").read()[-2500:])
        except OSError:
            continue
    fail(f"ウォームアップ失敗: {type(e).__name__}: {e}")

msg = (resp.get("choices") or [{}])[0].get("message") or {}
content = msg.get("content") or ""
reasoning = msg.get("reasoning_content") or ""
usage = resp.get("usage") or {}
ct = usage.get("completion_tokens") or 0
print(f"✅ 応答あり（{dt:.1f}s）")
print(f"   reply : {content[:120]!r}")
print(f"   think : {reasoning[:80]!r}")
print(f"   tokens: completion={ct} total={usage.get('total_tokens')}")
if ct:
    print(f"   速度  : {ct/dt:.1f} tokens/s（ウォームアップ込み）")
if not (content or reasoning or ct):
    fail("生成結果が空です（プロンプト/チャットテンプレートを確認）")
# 速度も残す。ログにしか無いと「前回より速い/遅い」を比べられない。
kc_save_timings({"speed_tps": round(ct / dt, 1) if ct and dt else None,
                 "warmup_seconds": round(dt, 1)})

# GPU 常駐（CPU フォールバックでも HTTP 200 は返るので必ず見る）
mem = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True).stdout.strip()
print(f"   VRAM  : {mem}")
# nvidia-smi の値は "N/A" や空になり得る。int() で落とさず、数字だけ拾う。
used = []
for _ln in mem.splitlines():
    _a = _ln.split(",")[0].strip()
    if _a.isdigit():
        used.append(int(_a))
if not used:
    # ここが旧実装の穴だった。`if used and ...` は **used が空だと素通り**するので、
    # nvidia-smi が何も返さない（= GPU 不在 / CPU 推論）ケースを検出できなかった。
    # 「エラーが出ずに黙って CPU 推論」は実測 0.9 t/s で、warning だけが手がかり。
    fail(f"nvidia-smi から VRAM を取得できません（mem={mem!r}）。CPU 実行の可能性。")
if sum(used) < 4000:
    fail(f"VRAM 使用量が少なすぎます（{used} MiB）。CPU 実行の可能性。")
kc_save_timings({"gpu_vram_mib": used})
