#!/usr/bin/env python3
"""cpu-race-probe: レースの稽古用カーネル（CPU・無料）を生成する。

GPU を使わずに `cli race` の全経路（push×2 → beacon → 判定 → delete →
勝ち側の継続 → url.txt 回収）を回すための偽カーネル。

probe の代わりに `KC_FAKE_MIB_S` を使う。side ごとに値を変えたいので
`KC_FAKE_MIB_<SIDE>` を優先して読む（ビルド時に確定させる）。

    KC_FAKE_MIB_A=200 KC_FAKE_MIB_B=20 \
      python3 -m cli race cpu-race-probe --gate 150 --yes
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from kbuild import metadata, pack_script, spec  # noqa: E402

OUT = HERE / "cpu-race-probe"


def _preamble():
    """ビルド時の env を `os.environ.setdefault` として焼く。

    `KC_FAKE_MIB_S` は side ごとに変えたいので、`KC_FAKE_MIB_<SIDE>`
    を優先して読む。env がなければ fallback。
    """
    side = os.environ.get("KC_PROBE_SIDE", "").strip()
    pairs = {
        "KC_PROBE_SIDE": side or "none",
        "KC_PROBE_NTFY_TOPIC": os.environ.get("KC_PROBE_NTFY_TOPIC", "").strip(),
        "KC_PROBE_GATE_MIB_S": os.environ.get("KC_PROBE_GATE_MIB_S", "0").strip(),
        "KC_FAKE_MIB_S": (os.environ.get("KC_FAKE_MIB_" + side.upper(), "").strip()
                          or os.environ.get("KC_FAKE_MIB_S", "").strip()
                          or "0"),
        "KC_FAKE_STARTUP_S": os.environ.get("KC_FAKE_STARTUP_S", "10").strip(),
    }
    lines = ["import os"]
    for k, v in pairs.items():
        if v:
            lines.append(f"os.environ.setdefault({k!r}, {v!r})")
    return "\n".join(lines) + "\n\n"


BODY = r'''#!/usr/bin/env python3
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
'''


META = metadata.for_kernel(spec.find("cpu-race-probe"), internet=True)

if __name__ == "__main__":
    pack_script.write(OUT, _preamble() + BODY, META, newline_meta=True)