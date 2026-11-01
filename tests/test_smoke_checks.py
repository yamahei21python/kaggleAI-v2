#!/usr/bin/env python3
"""smoke（エンジン直叩き）での疎通確認の判定を検証する。

実測（2026-10-31・#1 Ollama smoke）で、エンジンは正常なのに `[1] ui` だけが
FAIL して run 全体が ERROR になった。原因は「proxy が返すチャット UI」を
検査していたこと — smoke は proxy を立てないので `/` はエンジン自身の応答に
なり、Ollama は `"Ollama is running"`（17 B）を返す。

なので direct=True（proxy 無し）では [1] を **SKIP**（ok=None）にする。
SKIP は failures に数えないが PASS とも書かない、という契約も固定する。
"""
import pathlib
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (ソースのセル, エンジン名) — 3 エンジンとも同じ契約を持つ
CELLS = [
    (ROOT / "kaggle" / "step2" / "cells" / "06_checks.py", "ollama"),
    (ROOT / "kaggle" / "step2-vllm" / "cells" / "06_checks.py", "vllm"),
    (ROOT / "kaggle" / "llama-server" / "cells" / "06_checks.py", "llama-server"),
]


def _load(path, urlopen):
    """セルを exec して engine_checks を取り出す（MODEL_NAME 等はダミーで足りる）。"""
    ns = {"MODEL_NAME": "dummy", "PORT": 8000, "ALIAS": "dummy"}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    # セルは `import urllib.request` しているので、その名前を差し替える。
    # vLLM のセルは IPv4 強制の kc_urlopen_v4 を使うので、そちらも同じ stub にする。
    stub = types.SimpleNamespace(request=types.SimpleNamespace(urlopen=urlopen),
                                 error=__import__("urllib.error").error)
    ns["urllib"] = stub
    ns["kc_urlopen_v4"] = urlopen
    return ns["engine_checks"]


class _Resp:
    """urlopen の代わり。read() だけ真似る（with 文で使われる）。"""

    def __init__(self, body, status=200):
        self._body, self.status = body, status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _collect(engine_checks, **kw):
    calls = []

    def judge(idx, label, ok, detail):
        calls.append({"idx": idx, "label": label, "ok": ok, "detail": detail})

    def retry(fn, tries=6, delay=5):
        return fn()

    engine_checks(judge, retry, "http://127.0.0.1:8000", None, **kw)
    return calls


def _ui(calls):
    return next(c for c in calls if c["label"] == "ui")


def test_direct_mode_skips_ui():
    """direct=True なら [1] は SKIP（ok=None）で、HTTP を 1 回も叩かない。"""
    for path, name in CELLS:
        def boom(*a, **kw):
            raise AssertionError(f"{name}: direct なのに / を叩いた")
        calls = _collect(_load(path, boom), direct=True)
        ui = _ui(calls)
        assert ui["ok"] is None, f"{name}: direct では SKIP のはず（ok={ui['ok']}）"
        assert "skip" in ui["detail"], f"{name}: 理由が detail に無い: {ui['detail']}"


def test_proxy_mode_still_judges_ui():
    """direct=False（proxy 経由）では従来どおり判定する。

    Ollama の `/` は 17 B のテキストなので、proxy 経由でこれが返るのは異常。
    """
    for path, name in CELLS:
        calls = _collect(_load(path, lambda *a, **kw: _Resp(b"Ollama is running")),
                         direct=False)
        ui = _ui(calls)
        assert ui["ok"] is False, f"{name}: 17 B は FAIL のはず（ok={ui['ok']}）"
        assert "17B" in ui["detail"], f"{name}: サイズが detail に無い: {ui['detail']}"


def test_proxy_mode_passes_on_html():
    """proxy が 10 KB 超の HTML を返せば PASS（判定式そのものの回帰）。"""
    html = b"<!doctype html>\n" + b"x" * 12000
    for path, name in CELLS:
        calls = _collect(_load(path, lambda *a, **kw: _Resp(html)), direct=False)
        ui = _ui(calls)
        assert ui["ok"] is True, f"{name}: HTML は PASS のはず（ok={ui['ok']}）"


def test_judge_treats_none_as_skip_not_failure():
    """SKIP は failures に入らない（run を ERROR にしない）ことを固定する。

    `kbuild/serve_script.py` は `$SMOKE` などを埋め込むテンプレートなので、
    AST ではなく素の本文で順序を見る。
    """
    src = (ROOT / "kbuild" / "serve_script.py").read_text()
    assert "if ok is None:" in src, "judge に SKIP 分岐が無い"
    # ok=None の早期 return が failures.append より前に来ていること
    assert src.index("if ok is None:") < src.index("failures.append"), \
        "SKIP の return が failures.append より後ろにある"


def test_skeleton_engine_checks_accepts_every_argument_verify_passes():
    """骨格の**既定** `engine_checks` が、`verify()` と同じ引数を受けること。

    `checks_code` を持たないエンジン（exl3）はこの既定がそのまま使われる。
    既定だけ `auth=None` で止めていたため、2026-11-01 の exl3 実行が

        TypeError: engine_checks() takes from 3 to 4 positional arguments
                   but 5 were given

    で落ちた。**エンジンは起動していた**（`loaded in 229s; serving ... 9931`）ので、
    損失は 263.6 秒と 1 往復ぶんの GPU クォータまるごと。

    既存テストが 4 引数 + kwargs で呼んでいたため素通りしていた。ここでは
    「呼び出し側」と「既定の受け口」を**両方**固定して、片方だけ直る事故を防ぐ。
    """
    src = (ROOT / "kbuild" / "serve_script.py").read_text()

    assert "def engine_checks(judge, retry, base, auth=None, direct=False):" in src, (
        "骨格の既定 engine_checks が direct を受け取らない。"
        "checks_code を持たないエンジンが TypeError で落ちる。")

    assert "engine_checks(judge, retry, base, auth, direct)" in src, (
        "verify() が 5 引数で呼ばなくなっている。既定側の direct を外すなら"
        "呼び出し側も合わせること。")

    # 既定の定義が verify() の呼び出しより**前**にあること（後だと名前解決で落ちる）
    assert src.index("def engine_checks(judge, retry, base, auth=None, direct=False):") \
        < src.index("engine_checks(judge, retry, base, auth, direct)"), \
        "engine_checks の定義が verify() の呼び出しより後ろにある"


def test_engine_checks_without_checks_code_is_callable_with_five_args():
    """既定の engine_checks を実際に 5 引数で呼んで TypeError が出ないこと。

    上のテストは文字列一致なので、`direct=False` を消さずに別の理由で
    引数が減る変更（`*args` 化など）を見逃す。ここでは本文を取り出して
    exec し、本番と同じ 5 引数で叩く。
    """
    import re
    src = (ROOT / "kbuild" / "serve_script.py").read_text()
    m = re.search(r"^def engine_checks\(.*?(?=^def verify\()", src, re.S | re.M)
    assert m, "骨格の engine_checks を切り出せない"
    body = m.group(0)

    ns = {"ALIAS": "dummy", "data": {},
          # `/v1/models` は 1 引数、`/v1/chat/completions` は payload 込みの
          # 2 引数で呼ばれる。どちらも受ける。
          "http_json": lambda *a, **kw: (200, {"choices": [
              {"message": {"content": "2"}}]}),
          "print": lambda *a, **k: None}
    exec(compile(body, "serve_script:engine_checks", "exec"), ns)

    calls = []
    ns["engine_checks"](
        lambda idx, label, ok, detail: calls.append((label, ok)),
        lambda fn, tries=6, delay=5: fn(),
        "http://127.0.0.1:8000", None, True)      # ← 本番と同じ 5 引数
    assert [c[0] for c in calls] == ["models", "chat"], f"判定項目が違う: {calls}"
