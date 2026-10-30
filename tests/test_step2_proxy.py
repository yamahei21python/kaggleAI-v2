"""auth_proxy.py の実挙動テスト（実プロセスで起動する統合テスト）。

本体は `_step2_proxy_integration.py`。pkill / atexit でプロセスを後始末し、
モジュール読み込み時に一連の HTTP 検証を走らせる作りなので、pytest 本体と
同じプロセスに混ぜず、子プロセスとして実行する。
"""
import pathlib
import subprocess
import sys

SCRIPT = pathlib.Path(__file__).resolve().parent / "_step2_proxy_integration.py"


def _run():
    return subprocess.run([sys.executable, str(SCRIPT)],
                          capture_output=True, text=True)


def test_step2_proxy():
    # 実プロセス（pkill → 起動 → 停止）を扱うため、前回の終了が
    # 間に合わずポートが塞がる偽陰性がありうる。1 回だけ再試行する。
    r = _run()
    if r.returncode != 0:
        r = _run()
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
    assert r.returncode == 0, "auth_proxy 統合テスト失敗（上記ログ参照）"
