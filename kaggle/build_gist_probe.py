#!/usr/bin/env python3
"""gist-probe カーネル（CPU・無料）を生成する。

Gist への進捗書き込みが**実際に通るか**を、GPU を使わずに確かめる。

vLLM の起動待ちは 30 秒ごとに Gist へ状況を流す設計だが、2026-11-01 の
`vllm-playground` は Gist を 10/28 から一度も更新していなかった。理由
（トークン無し / 401 / 404）はどこにも残らず、外からは「何も起きていない」と
区別できなかった。GPU を回してから気づくと 30 分を捨てるので、CPU で
先に往復を確かめる。

実装は `runtime/gist.py`（本番と同じ）。ここで通れば本番も通る。
"""
import pathlib
import sys
from string import Template

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import metadata, pack_script, spec  # noqa: E402
from runtime import source_bundle  # noqa: E402

CFG = kcconfig.load(ROOT)
GIST_ID_DEFAULT = CFG.get("GIST_ID", "")

OWNER = "yamahei21python"
# CPU プローブの置き場は `kaggle/cpu-tests/`（README が台帳）。GPU を
# 1 秒も使わないので、モデルを落とす他のプローブと同じ棚に置く。
OUT = HERE / "cpu-tests" / "gist"

BODY = Template(r'''#!/usr/bin/env python3
"""Gist への書き込みを CPU だけで往復確認する（GPU 不要・無料）。

やること:
  1. GITHUB_TOKEN / GIST_ID が解決できるか
  2. PATCH で `vllm_status.txt` を書けるか
  3. GET で**読み返して**一致するか
  4. 存在しない Gist へ書いて、**失敗が失敗として返る**か（陰性対照）

2 までで満足しないのは、書けたつもりで別の Gist を見ている事故を分けるため。
4 を入れるのは、元の不具合が「失敗しても何も言わない」だったから——
成功だけ確かめても、失敗経路が沈黙したままなら同じ穴が残る。

失敗しても例外で落とさず result.json に理由を残す（見るのは人間なので、
スタックトレースより「どの段で何が返ったか」のほうが速い）。
"""
import json  # noqa: F811 — 上の束（runtime/gist.py）も import している
import os  # noqa: F811 — 同上。単体で読めるように本文にも残す
import sys
import traceback

WORK = "/kaggle/working"
RESULT = WORK + "/result.json"
STATUS_FILE = "vllm_status.txt"
# セル（03_vllm_serve.py）と同じ既定値。Secrets の GIST_ID は読まない——
# 本番が読んでいないものを読むと、probe だけ通って本番で落ちる。
GIST_ID_DEFAULT = "$GIST_ID"
# 誰の Gist でもない id。書き込みが 404 で**返る**ことの確認にだけ使う。
BOGUS_GIST = "0" * 32


def main():
    data = {"ok": False, "stage": "start", "python": sys.version.split()[0]}
    try:
        data["stage"] = "resolve"
        tok, tok_why = kc_gh_token_detail()
        gist_id = (os.environ.get("GIST_ID") or GIST_ID_DEFAULT or "").strip()
        data["token_present"] = bool(tok)
        data["token_detail"] = tok_why
        data["gist_id"] = gist_id
        print(f"token={'あり' if tok else 'なし'} ({tok_why}) gist_id={gist_id!r}")

        # ── 1〜3: 本番と同じ経路で往復する ──
        # `token=tok` を**渡さない**。空文字を明示すると selftest 側の
        # token_detail が「引数で明示」に潰れ、解決に失敗した理由
        # （ConnectionError = このカーネルに Secret が紐付いていない）が消える。
        data["stage"] = "selftest"
        r = kc_gist_selftest(gist_id, STATUS_FILE)
        data["selftest"] = r
        print("selftest: " + json.dumps(r, ensure_ascii=False))

        # ── 4: 陰性対照。失敗が理由つきで返らなければ、成功の判定も信用できない ──
        data["stage"] = "negative"
        nok, ndetail = kc_gist_write(BOGUS_GIST, STATUS_FILE, "probe\n",
                                     token=tok or None)
        data["negative"] = {"write_ok": nok, "detail": ndetail}
        # トークンが無い回は 1 で既に落ちているので、ここは「拒否されたか」を見る。
        data["negative_ok"] = (not nok) and bool(ndetail)
        print(f"negative: write_ok={nok} detail={ndetail!r}")

        data["ok"] = bool(r.get("ok")) and data["negative_ok"]
        if not data["ok"]:
            if not r.get("token_present"):
                data["advice"] = ("GITHUB_TOKEN を Secrets に登録し、"
                                  "**このカーネル（gist-probe）に紐付ける**。"
                                  "登録だけでは読めず、`ConnectionError` に"
                                  "なる（scope は gist でよい）")
            elif not r.get("write_ok"):
                data["advice"] = ("書き込みが拒否された。トークンの scope が "
                                  "gist か、GIST_ID が本人の Gist かを確認する")
            elif not r.get("read_ok") or not r.get("matched"):
                data["advice"] = ("書けたが読み返せない。GIST_ID が別の Gist を"
                                  "指している可能性がある（write_ok=True）")
            else:
                data["advice"] = ("往復は通ったが、失敗が失敗として返らない。"
                                  "kc_gist_write の例外処理を疑う")
            print("NG: " + data["advice"])
        else:
            print("OK: Gist の往復と失敗検出の両方が通った")
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
''').substitute(GIST_ID=GIST_ID_DEFAULT)

# 台帳から引く（手書きすると台帳とずれても気づけない）。
META = metadata.for_kernel(spec.find("gist-probe"), internet=True)

if __name__ == "__main__":
    pack_script.write(
        OUT,
        source_bundle("gist") + "\n\n" + BODY,
        META,
        newline_meta=True,
    )
