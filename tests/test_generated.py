"""コミット済みの生成物がビルダーの出力と一致するか確認する。

`kaggle` は PyPI の同名パッケージと衝突するため、kaggle/check_generated.py は
ファイルパスから読み込む（パッケージ import にしない）。

notebook と script の両方を見る（台帳の `kind` で分岐しない）。
"""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_generated", ROOT / "kaggle" / "check_generated.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_generated_matches_builders():
    assert _mod.main() == 0
