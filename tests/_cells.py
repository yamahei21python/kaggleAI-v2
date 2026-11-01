"""セル / テンプレートから関数を 1 つ切り出して exec する共通ヘルパー。

カーネルに埋め込まれるセル（`kaggle/**/cells/*.py`）は素の Python だが、
モジュール全体は未定義名（`data` や `kc_*`）を含むため丸ごとは exec できない。
テストに必要な関数だけを切り出して評価する。

**切り出し規則はここ 1 箇所だけ**に置く。以前は 3 つのテストファイルに
同じ実装が散っていて、片方だけ直すともう片方が古い規則のまま残る状態だった
（実際に「終端が `def` のみ」版と「インデント無し行」版が混在していた）。

`kbuild/serve_script.py` の `TEMPLATE` は `string.Template`（`$PLACEHOLDER` 入り）
なので、これも**置換前の文字列から**関数を切り出して exec する。プレースホルダは
関数本体に現れない範囲だけを切り出す（`def` 単位なら満たされる）。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def cell(*parts):
    """`kaggle/...` 配下のファイルを文字列で読む。

    例: `cell("step2-vllm", "cells", "03_vllm_serve.py")`
    """
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def slice_def(src, name):
    """`def name(` から 1 関数を切り出す。

    終端は「インデントの無い非空行」。`def` の直後の `X = 1` まで巻き込むと
    未定義名を評価して NameError になる（実測 2026-11-01）。
    関数内のネストした `def` はインデントされているので巻き込まない。
    """
    lines = src.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith(f"def {name}("))
    except StopIteration:
        raise AssertionError(f"関数 {name}() が見つからない") from None
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.strip() and not ln[:1].isspace():
            break
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"


def exec_def(src, name, **ns):
    """`slice_def` で切り出した関数を評価して返す（依存は `ns` で渡す）。"""
    exec(slice_def(src, name), ns)      # noqa: S102 - 自前ソースの切り出しのみ
    return ns[name]


def cell_def(*parts, name, **ns):
    """セルファイルから関数を 1 つ取り出す短縮形。"""
    return exec_def(cell(*parts), name, **ns)
