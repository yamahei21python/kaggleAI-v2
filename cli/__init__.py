"""ローカル CLI（build / push / check / list / ask / quota）。

    python3 -m cli                 # build（既定。push はしない）
    python3 -m cli build [name...]
    python3 -m cli push <name> --yes
    python3 -m cli list
    python3 -m cli check
    python3 -m cli ask "こんにちは"
    python3 -m cli quota

`push` は Kaggle 側で即実行され GPU クォータを消費するため、既定では走らせない。
`--yes` を付けたときだけ、残量を表示してから push する。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
