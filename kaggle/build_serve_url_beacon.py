#!/usr/bin/env python3
"""ビーコン（掲示板）カーネルの初期版を生成する。

**このカーネルは実行しない。** コードそのものが公開 URL の置き場で、
配信カーネルが `kernels_push` でこの `main.py` を書き換える。
読み手は `kaggle kernels pull` して `URL` / `STAGE` / `UPDATED` を読む。

    python3 kaggle/build_serve_url_beacon.py

生成物:
  kaggle/serve-url-beacon/         … 本番の掲示板（yamahei21python/serve-url-beacon）
  kaggle/cpu-tests/beacon-target/  … 検証用の掲示板（...-test）
                                     CPU プローブがここへ書く。**本番を汚さない**

なぜ 2 本か: 2026-11-01 に「古いテスト URL を掴む」事故が出ている。検証が
本番の掲示板を上書きすると同じ事故を自分で作ることになるので、置き場を分ける。
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from kbuild import metadata, pack_script  # noqa: E402
from runtime import beacon  # noqa: E402

OWNER = "yamahei21python"

# (出力先, slug, 説明) — 同じ中身を 2 か所に置く。
TARGETS = (
    (HERE / "serve-url-beacon", f"{OWNER}/serve-url-beacon",
     "本番の掲示板。配信カーネルが書き換える"),
    (HERE / "cpu-tests" / "beacon-target", f"{OWNER}/serve-url-beacon-test",
     "検証用。CPU プローブだけが書く（本番を汚さないため）"),
)


def main():
    for out, slug, why in TARGETS:
        body = beacon.kc_beacon_body("", stage="placeholder", updated=0.0)
        meta = metadata.base(
            id=slug,
            title=slug.rsplit("/", 1)[-1],
            code_file="main.py",
            kernel_type="script",
            gpu=False,
            datasets=[],
            # 掲示板は何も取りに行かない（中身はコードだけ）。
            internet=False,
        )
        pack_script.write(out, f"# {why}\n{body}", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
