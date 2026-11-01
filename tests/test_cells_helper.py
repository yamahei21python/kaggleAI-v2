"""`tests/_cells.py` の切り出し規則そのものを固定する。

ここが壊れると**全テストが別の関数を評価して緑になる**（偽の成功）ので、
ヘルパー自身にもテストを置く。2026-11-02 まで `def name(` で始まる行を
上から探していたため、同名の関数があると黙って先頭を選んでいた。
"""
import pytest

import _cells


SRC = '''
CONST = 1


def alpha(x):
    def inner():
        return 2
    return inner() + x


def beta():
    return "beta"
'''

DUPLICATED = SRC + '''

def alpha(y):
    return y
'''


def test_slice_def_returns_only_the_body():
    body = _cells.slice_def(SRC, "beta")
    assert body.strip() == 'def beta():\n    return "beta"'


def test_slice_def_keeps_nested_defs_inside():
    body = _cells.slice_def(SRC, "alpha")
    assert "def inner():" in body, "ネストした def を切り落としている"
    assert "return inner() + x" in body


def test_slice_def_stops_at_the_next_top_level_def():
    body = _cells.slice_def(SRC, "alpha")
    assert "beta" not in body, "次の関数まで巻き込んでいる"


def test_slice_def_refuses_ambiguous_names():
    """同名の def が 2 つあったら例外（先頭を黙って選ばない）。"""
    with pytest.raises(AssertionError) as e:
        _cells.slice_def(DUPLICATED, "alpha")
    assert "2 個ある" in str(e.value)


def test_slice_def_reports_missing_names():
    with pytest.raises(AssertionError) as e:
        _cells.slice_def(SRC, "gamma")
    assert "見つからない" in str(e.value)


def test_slice_def_works_on_templates_with_dollar_tokens():
    """`$NAME` 入りのテンプレートでも切り出せる（潰してから構文解析する）。"""
    from kbuild import serve_script
    body = _cells.slice_def(serve_script.TEMPLATE.template, "stage_env")
    assert body.startswith("def stage_env(")
    # `$ENGINE` / `$MODE` を含む f-string がそのまま切り出せている
    assert "{ENGINE} serve ({MODE})" in body


def test_exec_def_evaluates_with_the_given_namespace():
    assert _cells.exec_def(SRC, "beta")() == "beta"
