import os
os.environ.setdefault('KC_PROBE_SIDE', 'none')
os.environ.setdefault('KC_PROBE_GATE_MIB_S', '0')
os.environ.setdefault('KC_FAKE_MIB_S', '0')
os.environ.setdefault('KC_FAKE_STARTUP_S', '10')

#!/usr/bin/env python3
"""レース稽古用。probe の代わりに焼いた偽の速度を beacon へ流す。

- `KC_FAKE_MIB_S` 未満なら即終了（ゲートと同じ）
- 通ったら `KC_FAKE_STARTUP_S` 秒 sleep して url.txt を書く（install の代役）
"""
import json
import os
import time
import urllib.request

SIDE = os.environ.get("KC_PROBE_SIDE", "").strip()
TOPIC = os.environ.get("KC_PROBE_NTFY_TOPIC", "").strip()
MIB = float(os.environ.get("KC_FAKE_MIB_S", "0") or 0)
GATE = float(os.environ.get("KC_PROBE_GATE_MIB_S", "0") or 0)
STARTUP = float(os.environ.get("KC_FAKE_STARTUP_S", "60") or 60)


def post(msg):
    if not TOPIC:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(
            "https://ntfy.sh/" + TOPIC, data=msg.encode(), method="POST"),
            timeout=10).read()
    except Exception as e:
        print("beacon fail: %s: %s" % (type(e).__name__, e), flush=True)


print("side=%s fake_mib=%s gate=%s" % (SIDE, MIB, GATE), flush=True)
post("mib_s=%s side=%s" % (MIB, SIDE))

if GATE > 0 and MIB < GATE:
    print("gate reject（install の代役は走らせない）", flush=True)
    raise SystemExit(0)

print("continue: startup %ss" % STARTUP, flush=True)
step = max(5.0, STARTUP / 6.0)
for i in range(6):
    time.sleep(step)
    print("startup %d/6" % (i + 1), flush=True)

url = "https://%s-fake.trycloudflare.com" % (SIDE or "x")
with open("url.txt", "w", encoding="utf-8") as f:
    f.write(url)
with open("result.json", "w", encoding="utf-8") as f:
    json.dump({"url": url, "side": SIDE, "mib_s": MIB}, f)
post("mib_s=%s side=%s" % (MIB, SIDE))
post("url=%s side=%s" % (url, SIDE))
print("URL " + url, flush=True)
