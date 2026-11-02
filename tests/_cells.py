"""セル / テンプレートから関数を 1 つ切り出して exec する共通ヘルパー。

カーネルに埋め込まれるセル（`kaggle/**/cells/*.py`）は素の Python だが、
モジュール全体は未定義名（`data` や `kc_*`）を含むため丸ごとは exec できない。
テストに必要な関数だけを切り出して評価する。

**切り出し規則はここ 1 箇所だけ**に置く。以前は 3 つのテストファイルに
同じ実装が散っていて、片方だけ直すともう片方が古い規則のまま残る状態だった
（実際に「終端が `def` のみ」版と「インデント無し行」版が混在していた）。

`kbuild/serve_script.py` の `TEMPLATE` は `string.Template`（`$PLACEHOLDER` 入り）
なので、`$NAME` を名前だけに潰してから構文解析する（行番号は変わらないので、
切り出しは元の文字列から行う）。

切り出しは `ast` で行う。以前は「`def name(` で始まる行を上から探す」だった
ので、同名の関数が 2 つあると**黙って先頭を選んでいた**（2026-11-02 に
`reason = loop(procs)` が 2 箇所あり、位置比較のテストが誤った箇所を掴んだ）。
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# `string.Template` の `$NAME` / `${NAME}`。潰すのは名前だけ（`$` を消す）。
_TOKEN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def cell(*parts):
    """`kaggle/...` 配下のファイルを文字列で読む。

    例: `cell("step2-vllm", "cells", "03_vllm_serve.py")`
    """
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _neutralize(src):
    """`$NAME` を `NAME` にして構文解析できるようにする（行数・行番号は不変）。"""
    return _TOKEN.sub(lambda m: m.group(1) or m.group(2), src)


def slice_def(src, name):
    """`def name(` から 1 関数を切り出す（本体だけ・デコレータは含めない）。

    同名の関数が複数あるときは**例外**にする。黙って先頭を選ぶと、
    テストが別の関数を評価して緑になる（偽の成功）。
    """
    tree = ast.parse(_neutralize(src))
    found = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == name]
    if not found:
        raise AssertionError(f"関数 {name}() が見つからない")
    if len(found) > 1:
        lines = [f"  {n.lineno} 行目" for n in found]
        raise AssertionError(
            f"関数 {name}() が {len(found)} 個ある（切り出し先を決められない）:\n"
            + "\n".join(lines))
    node = found[0]
    lines = src.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno]) + "\n"


def exec_def(src, name, **ns):
    """`slice_def` で切り出した関数を評価して返す（依存は `ns` で渡す）。"""
    exec(slice_def(src, name), ns)      # noqa: S102 - 自前ソースの切り出しのみ
    return ns[name]
