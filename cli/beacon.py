"""ビーコン（掲示板カーネル）から公開 URL を読む。

配信カーネルは `kernels_push` で掲示板の `main.py` を書き換える。こちらは
`kaggle kernels pull` でそれを引き、`URL` / `STAGE` / `UPDATED` を読む。

    python3 -m cli beacon              # 既定の掲示板を読む
    python3 -m cli beacon --slug yamahei21python/serve-url-beacon
    python3 -m cli beacon --json       # 機械可読（スクリプトから使う）

**古い値を掴まないこと**がこのコマンドの主目的。2026-11-01 にテスト用の URL を
本物として読む事故が出ているので、既定で
  - `STAGE == "probe"` の値は捨てる
  - `--max-age` 秒より古い値は捨てる
の 2 つを必ず適用する。捨てたときは理由を出す（黙って古い URL を返さない）。
"""
import argparse
import json
import sys
import time

from kbuild import spec

DEFAULT_NAME = "serve-url-beacon"


def _slug(arg=""):
    if arg:
        return arg
    return f"{spec.OWNER}/{DEFAULT_NAME}"


def _kaggle_exe():
    """`kaggle` の解決順は push と同じにする（KC_KAGGLE → リポジトリ内 venv →
    PATH）。この python に kaggle パッケージが入っていなくても CLI で引ける。"""
    try:
        from cli.push import kaggle_exe
        return kaggle_exe()
    except Exception:                     # 解決に失敗しても PATH で試す
        return ""


def _age(data, now=None):
    """掲示板の値が何秒前か。`UPDATED` が無ければ None。"""
    if not data.get("updated"):
        return None
    return (time.time() if now is None else now) - data["updated"]


def judge(data, max_age, allow_probe=False):
    """その値を本物として使えるか。使えないなら**理由**（使えるなら空文字）。

    捨てる条件を 1 箇所に置く。`cli beacon`（表示）と `cli ask`（URL 解決）が
    別々に判定すると、片方だけ直したときに「表示は ok なのに ask が古い値を
    掴む」のような食い違いが出る。2026-11-01 に古いテスト URL を本物として
    読む事故をしているので、条件は増やすときもここだけ。
    """
    if not data.get("url"):
        return "URL が空（まだ配信が無い / placeholder）"
    if data.get("stage") == "probe" and not allow_probe:
        return "STAGE=probe（検証用の値なので捨てた。--allow-probe で受け入れる）"
    age = _age(data)
    if age is not None and age > max_age:
        return f"{int(age)} 秒前の値（--max-age {max_age} を超えている）"
    return ""


def resolve_url(slug="", max_age=6 * 3600, allow_probe=False):
    """ビーコンから公開 URL を決める。戻り値は `(url, why)`。

    `url` が空のとき `why` に理由が入る（引けなかったのか / 捨てたのか）。
    クライアント（`cli ask`）はここを通す。判定は `judge()` と共通。
    """
    from runtime import beacon

    s = _slug(slug)
    ok, detail, data = beacon.kc_beacon_pull(s, exe=_kaggle_exe())
    if not ok:
        return "", f"{s} を引けない（{detail}）"
    why = judge(data, max_age, allow_probe)
    if why:
        return "", why
    return data["url"].rstrip("/"), ""


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cli beacon", description=__doc__)
    ap.add_argument("--slug", default="", help="owner/name（省略時は既定）")
    ap.add_argument("--json", action="store_true", help="JSON で出す")
    ap.add_argument("--max-age", type=int, default=6 * 3600,
                    help="この秒数より古い値は捨てる（既定 6 時間）")
    ap.add_argument("--allow-probe", action="store_true",
                    help="STAGE=probe も受け入れる（検証用）")
    a = ap.parse_args(argv)

    # runtime/beacon.py をそのまま使う（カーネル内と**同じパーサ**で読む）。
    from runtime import beacon

    slug = _slug(a.slug)
    ok, detail, data = beacon.kc_beacon_pull(slug, exe=_kaggle_exe())
    if not ok:
        print(f"✗ {slug}: {detail}", file=sys.stderr)
        return 2

    age = _age(data)
    why = judge(data, a.max_age, a.allow_probe)

    if a.json:
        print(json.dumps({"slug": slug, "ok": not why, "why": why,
                          "age_seconds": None if age is None else round(age, 1),
                          **data}, ensure_ascii=False, indent=2))
    else:
        print(f"📡 {slug}  ({detail})")
        if data.get("url"):
            print(f"   URL    : {data['url']}")
        print(f"   STAGE  : {data.get('stage') or '-'}")
        if age is None:
            print("   UPDATED: -")
        else:
            print(f"   UPDATED: {int(age)} 秒前 ({data['updated']:.0f})")
        if why:
            print(f"   ⚠ {why}")
    return 0 if not why else 1


if __name__ == "__main__":
    sys.exit(main())
