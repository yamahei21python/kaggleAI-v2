"""設定セル（`kaggle/*/cells/00_settings.py`）の __TOKEN__ 置換。

同じ設定を notebook 版と script 版の**両方**が読むための共通処理。
片方だけ直して片方が古い、という事故（`OLLAMA_BIN` の定義漏れ）を
構造的に防ぐのが目的。

置換は「長いトークンから」行う。`__MODEL__` は `__MODEL_DISPLAY__` の
前方部分と一致するため、短い方から置換すると壊れる。
"""
import pathlib


def render(path, tokens: dict) -> str:
    text = pathlib.Path(path).read_text(encoding="utf-8")
    for key in sorted(tokens, key=len, reverse=True):
        text = text.replace(key, str(tokens[key]))
    leftover = [k for k in tokens if k in text]
    if leftover:
        raise ValueError(
            f"{pathlib.Path(path).name}: 未置換のトークン {leftover}")
    return text
