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

    # `kaggle` の解決順は push と同じにする（KC_KAGGLE → リポジトリ内 venv →
    # PATH）。この python に kaggle パッケージが入っていなくても CLI で引ける。
    try:
        from cli.push import kaggle_exe
        exe = kaggle_exe()
    except Exception:                     # 解決に失敗しても PATH で試す
        exe = ""

    slug = _slug(a.slug)
    ok, detail, data = beacon.kc_beacon_pull(slug, exe=exe)
    if not ok:
        print(f"✗ {slug}: {detail}", file=sys.stderr)
        return 2

    now = time.time()
    age = (now - data["updated"]) if data.get("updated") else None
    why = ""
    if not data.get("url"):
        why = "URL が空（まだ配信が無い / placeholder）"
    elif data.get("stage") == "probe" and not a.allow_probe:
        why = "STAGE=probe（検証用の値なので捨てた。--allow-probe で受け入れる）"
    elif age is not None and age > a.max_age:
        why = f"{int(age)} 秒前の値（--max-age {a.max_age} を超えている）"

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
