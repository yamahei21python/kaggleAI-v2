"""コミット済みノートブックがビルダーの出力と一致するか確認する。

`kaggle` は PyPI の同名パッケージと衝突するため、kaggle/check_notebooks.py は
ファイルパスから読み込む（パッケージ import にしない）。
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_notebooks", ROOT / "kaggle" / "check_notebooks.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_notebooks_match_builders():
    assert _mod.main() == 0
