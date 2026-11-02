"""生成物に `__NAME__` のプレースホルダが残っていないかを見る。

ビルダーはセル中の `__MODEL__` のようなトークンを `str.replace` で埋める。
**置換漏れは静かに通る** — 生成物の `MODEL_NAME = "__MODEL__"` はビルドも
構文検査も通り、GPU を掴んだあとに初めて壊れる。実害の例:

    KAGGLE_OWNER = "__KAGGLE_OWNER__"   # ビーコンの slug が不正になる
    MODEL_SRC = kc_find_dataset("__DATASET_SLUG__")   # 入力が無いと言われる

script 側は `serve_script` の `$` トークン検査（`_check_tokens`）が双方向で
同じ役目を果たしている。ここは **notebook と連結 script の受け皿**で、
`pack_nb` / `pack_script` の書き出し口で必ず通す。

`__all__` / `__init__` のような小文字の dunder は対象外（大文字始まりのみ）。
"""
import re

TOKEN = re.compile(r"__([A-Z][A-Z0-9_]{0,30})__")


def leftovers(text):
    """残っている `__NAME__` を名前だけ昇順で返す。"""
    return sorted(set(TOKEN.findall(text or "")))


def check(text, where=""):
    """残っていれば `ValueError`。戻り値はそのまま返す（呼び出し側で使える）。"""
    bad = leftovers(text)
    if bad:
        raise ValueError(
            "{}: プレースホルダが残っている {} — ビルダーの置換漏れ。"
            "セル側で名前を変えたらビルダーの置換も直す".format(
                where or "生成物", ", ".join("__" + b + "__" for b in bad)))
    return text
