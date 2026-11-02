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
    "quota": "quota.py",
    # 公開 URL の置き場。カーネル自体を掲示板にする（Secrets 不要）。
    # 経緯は notes/url-publishing.md。
    "beacon": "beacon.py",
    "ollama": "engines/ollama.py",
    "lmdeploy": "engines/lmdeploy.py",
    "sglang": "engines/sglang.py",
}

HEADER = (
    "# ============================================================\n"
    "# 共通ランタイム（runtime/*.py から自動生成。手で編集しない）\n"
    "# proxy / tunnel / keepalive / verify の配管をここに集約する。\n"
    "# ============================================================\n"
)


def module_source(name):
    """1 モジュールのソースだけ（HEADER 無し）。

    骨格（`kbuild/serve_script.py` / `serve_notebook.py`）が、呼び出し側から
    渡された束に**後から 1 本だけ足す**ときに使う。HEADER を二重にしない。
    """
    rel = MODULES[name]
    body = (HERE / rel).read_text(encoding="utf-8").rstrip()
    return f"# ── runtime/{rel} " + "─" * max(0, 46 - len(rel)) + "\n" + body + "\n\n"


def source_bundle(*names):
    """指定モジュールのソースを連結した 1 セル分の文字列を返す。"""
    parts = [HEADER]
    for name in names:
        parts.append(module_source(name))
    return "".join(parts)


def proxy_source(path):
    """auth_proxy.py を単体実行できる形にする。

    proxy はカーネル上で単一ファイルとして走るため import できない。
    クォータ実装（runtime/quota.py）を先頭に同梱し、proxy 側は
    kc_* が未定義のときだけローカルの runtime を import する。
    """
    quota = (HERE / MODULES["quota"]).read_text(encoding="utf-8").rstrip()
    body = pathlib.Path(path).read_text(encoding="utf-8")
    return (f"# ── runtime/{MODULES['quota']}（auth_proxy.py に同梱）\n"
            f"{quota}\n\n{body}")
