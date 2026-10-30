"""カーネル生成コードへ埋め込む共通ランタイム（真実源）。

build_*.py は source_bundle() でこのパッケージのソースを生成物へ連結する。
Kaggle 上で runtime パッケージとして import されることはない。

モジュールを増やすときは MODULES に登録する。生成コードは notebook の
セル間で名前を共有するため、関数名は kc_ 接頭辞で衝突を避ける。
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

MODULES = {
    "net": "net.py",
    "proc": "proc.py",
    "storage": "storage.py",
    "keepalive": "keepalive.py",
    "proxy": "proxy.py",
    "ollama": "engines/ollama.py",
}

HEADER = (
    "# ============================================================\n"
    "# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）\n"
    "# proxy / tunnel / keepalive / verify の配管をここに集約する。\n"
    "# ============================================================\n"
)


def source_bundle(*names):
    """指定モジュールのソースを連結した 1 セル分の文字列を返す。"""
    parts = [HEADER]
    for name in names:
        rel = MODULES[name]
        parts.append(f"# ── runtime/{rel} " + "─" * max(0, 46 - len(rel)) + "\n")
        parts.append((HERE / rel).read_text(encoding="utf-8").rstrip() + "\n\n")
    return "".join(parts)
