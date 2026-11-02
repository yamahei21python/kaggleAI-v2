#!/usr/bin/env python3
"""beacon-probe カーネル（CPU・無料）を生成する。

ビーコン方式（カーネルのコードを掲示板にする）が**実際に往復するか**を、
GPU を使わずに確かめる。

確かめること:
  1. カーネル内から Kaggle の認証が取れるか（Secrets ではなく注入認証）
  2. `kernels_push` で掲示板を書けるか
  3. `kernels_pull` で**読み返して**一致するか
  4. `UPDATED` の鮮度判定が古い値を弾くか（テスト URL を掴む事故の防波堤）
  5. 失敗が理由つきで返るか（空 slug を渡して陰性対照）

書き込み先は **検証用の掲示板**（`.../serve-url-beacon-test`）。
本番の `serve-url-beacon` は触らない——検証が本番を上書きすると、
2026-11-01 に踏んだ「古いテスト URL を掴む」事故を自分で作ることになる。

    python3 kaggle/build_beacon_probe.py
"""
import pathlib
import sys
from string import Template

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from kbuild import metadata, pack_script, spec  # noqa: E402
from runtime import source_bundle  # noqa: E402

OWNER = "yamahei21python"
OUT = HERE / "cpu-tests" / "beacon"
TARGET_SLUG = f"{OWNER}/serve-url-beacon-test"

BODY = Template(r'''#!/usr/bin/env python3
"""ビーコンの往復を CPU だけで確かめる（GPU 不要・無料）。

書く → 読み返す → 鮮度判定、までを 1 回で見る。失敗しても例外で落とさず
result.json に理由を残す（見るのは人間なので、スタックトレースより
「どの段で何が返ったか」のほうが速い）。
"""
import json  # noqa: F811 — 上の束も import している
import os  # noqa: F811 — 同上。単体で読めるように本文にも残す
import sys  # noqa: F811 — runtime/beacon.py が kc_kaggle_path で使う
import time  # noqa: F811 — 上の束（runtime/beacon.py）も import している
import traceback

WORK = "/kaggle/working"
RESULT = WORK + "/result.json"
SLUG = "$TARGET_SLUG"
PROBE_URL = "https://beacon-probe.invalid"


def main():
    data = {"ok": False, "stage": "start", "python": sys.version.split()[0],
            "slug": SLUG}
    try:
        # ── 1〜3: 書いて、読み返す ──
        data["stage"] = "publish"
        t0 = time.time()
        ok, detail = kc_beacon_publish(PROBE_URL, stage="probe", slug=SLUG)
        data["publish"] = {"ok": ok, "detail": detail}
        print(f"publish: ok={ok} {detail}")
        if not ok:
            data["advice"] = ("push できない。カーネル内の注入認証が効いているか、"
                              "slug の owner がこのカーネルの持ち主かを確認する")
            return 1

        data["stage"] = "pull"
        ok, detail, got = kc_beacon_pull(SLUG)
        data["pull"] = {"ok": ok, "detail": detail, "data": got}
        print(f"pull: ok={ok} {detail} {json.dumps(got, ensure_ascii=False)}")
        if not ok:
            data["advice"] = "push は通ったのに pull が失敗した（権限か反映待ち）"
            return 1

        # ── 4: 中身と鮮度 ──
        data["stage"] = "verify"
        data["url_matches"] = got.get("url") == PROBE_URL
        data["stage_matches"] = got.get("stage") == "probe"
        # push の**前**のスタンプで判定 → 本物。1 時間後で判定 → 偽物。
        data["fresh_now"] = kc_beacon_is_fresh(got, t0)
        data["fresh_later"] = kc_beacon_is_fresh(got, t0 + 3600)
        print(f"url一致={data['url_matches']} stage一致={data['stage_matches']}"
              f" 鮮度(now)={data['fresh_now']} 鮮度(+1h)={data['fresh_later']}")

        # ── 5: 陰性対照。失敗が理由つきで返らなければ、成功判定も信用できない ──
        data["stage"] = "negative"
        nok, ndetail = kc_beacon_publish("x", slug="")
        data["negative"] = {"ok": nok, "detail": ndetail}
        data["negative_ok"] = (not nok) and bool(ndetail)
        print(f"negative: ok={nok} {ndetail}")

        data["ok"] = (data["url_matches"] and data["stage_matches"]
                      and data["fresh_now"] and not data["fresh_later"]
                      and data["negative_ok"])
        if not data["ok"]:
            data["advice"] = "往復はしたが判定が合わない（鮮度・stage の実装を疑う）"
        else:
            print("OK: ビーコンの往復と鮮度判定が通った")
    except Exception as e:
        data["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        os.makedirs(WORK, exist_ok=True)
        with open(RESULT, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("RESULT: " + json.dumps(data, ensure_ascii=False))
    return 0 if data.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
''').substitute(TARGET_SLUG=TARGET_SLUG)

# 台帳から引く（id / title / gpu / datasets を手書きすると、台帳とずれても
# 気づけない。`cli build` は台帳の env を渡すだけで、出力はビルダーが書いた値）。
META = metadata.for_kernel(
    spec.find("beacon-probe"),
    # Kaggle API を叩くのでインターネットは要る（GPU は使わない）。
    internet=True,
)

if __name__ == "__main__":
    pack_script.write(
        OUT,
        source_bundle("beacon") + "\n\n" + BODY,
        META,
        newline_meta=True,
    )
