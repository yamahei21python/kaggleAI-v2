"""pytest 共通設定。リポジトリ直下と `tests/` を import パスに追加する。

`tests/` を足すのは、テスト用ヘルパー `tests/_cells.py` を `import _cells` で
読むため。`from tests import _cells` は **site-packages の `tests` パッケージ**に
吸われて ImportError になる（実測 2026-11-01）。
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
