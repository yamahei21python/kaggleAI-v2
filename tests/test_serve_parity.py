"""配信 2 実装（notebook セル / script 骨格）の**振る舞い**の parity。

`notes/serve-parity.md` の表のうち、script 版に無かった 3 つを固定する。

| 機能 | notebook 版 | script 版（このテストが守る） |
|---|---|---|
| トンネル監視 | 10 分ごとに `/healthz` | `loop()` が `TUNNEL_CHECK_EVERY` 周ごとに叩く |
| トンネル再起動 | 死んだら 1 回だけ + URL 再公開 | `restart_tunnel()` を 1 回だけ |
| 証跡 | `keepalive.txt` / `cleanup.txt` | `save_evidence()` が同じ名前で出す |

関数の同一性ではなく**振る舞い**で書く（制御の流れは構造的に共有できない。
`notes/serve-parity.md` の「なぜ 1 本にしていないか」）。
"""
import pathlib

import pytest

import _cells
from kbuild import serve_script

NOTEBOOK_KEEPALIVE = ("kaggle", "serve-cells", "keepalive.py")


def _template():
    return serve_script.TEMPLATE.template


class _Clock:
    """`time` の代わり。`sleep` だけが時間を進める（実時間を待たない）。"""

    def __init__(self, step=20):
        self.t = 0.0
        self.step = step
        self.slept = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += self.step

    def strftime(self, fmt):
        return "00:00:00"


def _loop_ns(clock, *, minutes=1, restart=None, health=None,
             check_every=30, tunnel="cloudflared", revive=True):
    """`loop()` を評価するための名前空間。外部との接点はここだけ。

    `revive=True` は本物の `restart_tunnel()` と同じく、張り直したプロセスを
    `procs` に差し替える（差し替えないと次の周で「まだ死んでいる」になる）。
    """
    calls = {"restart": 0, "health": 0}

    def _restart(procs):
        calls["restart"] += 1
        new_url = restart(procs) if restart else "https://new.trycloudflare.com"
        if new_url:
            procs[tunnel] = _Proc(None if revive else 1)
        return new_url

    def _health(url):
        calls["health"] += 1
        return health(url) if health else (True, "HTTP200")

    lines = []
    ns = {
        "time": clock,
        "print": lambda *a, **k: None,
        "CONF_DRIVEN": False,          # conf ではなく分数で終わらせる
        "RUN_MINUTES": minutes,
        "HARD_MAX_MINUTES": 10 ** 6,
        "read_conf": lambda: (1, minutes, 0.0),
        "kc_keepalive_expired": lambda *a, **k: False,
        "TUNNEL": tunnel,
        "LOOP_SLEEP": 20,
        "TUNNEL_CHECK_EVERY": check_every,
        "restart_tunnel": _restart,
        "_public_health": _health,
        "kc_proc_state": lambda name, match="exact": "alive",
        "kc_gpu_usage": lambda: "0, 0",
        "KEEPALIVE_LINES": lines,
        "data": {},
        "calls": calls,
    }
    fn = _cells.exec_def(_template(), "loop", **ns)
    return fn, ns, calls


class _Proc:
    """`poll()` と `returncode` だけを持つプロセスの代役。"""

    def __init__(self, code=None):
        self.code = code
        self.returncode = code

    def poll(self):
        return self.code


def test_loop_restarts_a_dead_tunnel_once_and_keeps_the_new_url():
    """トンネルが死んだら **1 回だけ**張り直し、新しい URL で監視を続ける。

    notebook 版（`serve-cells/keepalive.py`）の `tunnel_restarted == 0` と同じ。
    張り直した URL を `data["url"]` に残さないと、成果物と Gist / ビーコンが
    失効 URL を指したままになる。
    """
    clock = _Clock()
    fn, ns, calls = _loop_ns(clock, minutes=2)
    reason = fn({"cloudflared": _Proc(1)}, "")

    assert reason == "budget", "張り直した後は止まらずに走り続けること"
    assert calls["restart"] == 1, "再起動は 1 回だけ"
    assert ns["data"]["url"] == "https://new.trycloudflare.com"
    assert ns["KEEPALIVE_LINES"], "毎周期の控えが残っていない"


def test_loop_gives_up_when_the_restart_fails():
    """張り直しに失敗したら、その場で終わる（無限に起こし続けない）。"""
    clock = _Clock()
    fn, _ns, calls = _loop_ns(clock, minutes=5, restart=lambda procs: "")
    reason = fn({"cloudflared": _Proc(1)}, "")

    assert reason == "cloudflared_exited(1)"
    assert calls["restart"] == 1


def test_loop_gives_up_on_the_second_tunnel_death():
    """2 回目は起こさない。同じ原因で落ち続けるなら URL は生き残らない。"""
    clock = _Clock()
    # 張り直したプロセスもすぐ死ぬ（revive=False）→ 次の周で 2 回目の死
    fn, _ns, calls = _loop_ns(clock, minutes=5, revive=False)
    reason = fn({"cloudflared": _Proc(1)}, "")

    assert reason == "cloudflared_exited(1)"
    assert calls["restart"] == 1


def test_loop_counts_public_health_failures():
    """`/healthz` の失敗を数える。pid が生きていても URL は失効しうる。

    notebook 版は 2 回で fail する（`tunnel_dead >= 2`）。script 版も
    `data["tunnel_dead"]` に同じ数を残し、`run()` が同じ判定をする。
    """
    clock = _Clock()
    fn, ns, calls = _loop_ns(
        clock, minutes=1, check_every=1,
        health=lambda url: (False, "TimeoutError"))
    reason = fn({"cloudflared": _Proc(None)}, "https://x.trycloudflare.com")

    assert reason == "budget"
    assert calls["health"] == 3, "毎周チェックしていない"
    assert ns["data"]["tunnel_dead"] == 3
    assert any("tunnel=DEAD" in ln for ln in ns["KEEPALIVE_LINES"])


def test_loop_resets_the_dead_count_after_a_restart():
    """張り直した直後は 0 に戻す（別のトンネルなので旧カウントを混ぜない）。

    notebook 版の `tunnel_dead = 0` / `_restarted_now = True` と同じ。
    混ぜると、稼働中でも次のチェックで偽の失敗になる。
    """
    clock = _Clock()
    ns_health = {"dead": True}

    def health(url):
        return (False, "TimeoutError") if ns_health["dead"] else (True, "HTTP200")

    fn, ns, _calls = _loop_ns(clock, minutes=4, check_every=1, health=health)
    # 1 周目: 死んでいる → 張り直し → dead=0、その周は判定しない
    reason = fn({"cloudflared": _Proc(1)}, "https://x.trycloudflare.com")

    assert reason == "budget"
    assert ns["data"]["tunnel_dead"] >= 1
    # 再起動の周だけはチェックしていない（_restarted_now で飛ばす）
    assert ns["KEEPALIVE_LINES"][0].count("tunnel=") >= 1


def test_save_evidence_writes_the_same_files_as_the_notebook(tmp_path):
    """`keepalive.txt` / `cleanup.txt` を notebook 版と同じ名前で出す。

    これが無いと、外から繋がらなかった回に「いつ URL が死んだか」を
    成果物から読めない（実測 2026-11-02 の v16 は result.json だけだった）。
    """
    ns = {
        "print": lambda *a, **k: None,
        "KEEPALIVE_LINES": ["[00:00:01] tunnel=cloudflared:alive | tunnel=OK"],
        "data": {"stopped_by": "budget", "tunnel_dead": 0,
                 "tunnel_restarted": False, "url": "https://x",
                 "engine_cleanup": "unloaded", "stop_remaining": []},
        "TUNNEL": "cloudflared",
        "SERVE": True,
        "KEEPALIVE_LOG": str(tmp_path / "keepalive.txt"),
        "CLEANUP_LOG": str(tmp_path / "cleanup.txt"),
    }
    _cells.exec_def(_template(), "save_evidence", **ns)()

    keep = (tmp_path / "keepalive.txt").read_text(encoding="utf-8")
    clean = (tmp_path / "cleanup.txt").read_text(encoding="utf-8")
    assert "stopped_by=budget" in keep
    assert "tunnel_dead=0" in keep
    assert "tunnel=cloudflared:alive" in keep
    assert clean.splitlines() == ["engine_cleanup=unloaded", "leftover=none"]


def test_script_fails_when_the_url_stayed_dead():
    """`/healthz` が 2 回以上失敗したままなら `ok: true` にしない。

    ここが今回の穴だった: URL が失効しても `result.json` は `ok: true` で、
    外からは繋がらないまま COMPLETE した。
    """
    src = _template()
    i = src.index('if data.get("tunnel_dead", 0) >= 2:')
    block = src[i:i + 700]
    assert "raise RuntimeError" in block
    assert "keepalive.txt" in block
    # 判定は loop() の後（ループの中で握りつぶさない）
    assert src.index("reason = loop(procs, url)") < i


def test_both_implementations_monitor_the_tunnel():
    """notebook 側と script 側が**同じ 3 つ**を持っていること。

    片方だけ直すと parity が崩れるので、名前ではなく役割で突き合わせる。
    """
    nb = _cells.cell(*NOTEBOOK_KEEPALIVE)
    sc = _template()
    # 1) 定期チェック
    assert "TUNNEL_CHECK_EVERY" in nb and "TUNNEL_CHECK_EVERY" in sc
    # 2) 1 回だけの再起動
    assert "tunnel_restarted == 0" in nb
    assert "if restarted:" in sc
    # 3) 証跡
    assert 'save_output("keepalive.txt"' in nb
    assert "KEEPALIVE_LOG" in sc
    assert 'save_output("cleanup.txt"' in nb
    assert "CLEANUP_LOG" in sc


def test_script_tunnel_check_runs_every_ten_minutes():
    """監視の周期が notebook 版（10 分）と揃っていること。

    周期そのものが違うと「同じ監視」と言えない。script のループは 20 秒なので
    30 周 = 600 秒。
    """
    src = _template()
    assert "TUNNEL_CHECK_EVERY = 600 // LOOP_SLEEP" in src
    assert "LOOP_SLEEP = 20" in src
    nb_settings = _cells.cell("kaggle", "step2", "cells", "00_settings.py")
    assert "TUNNEL_CHECK_EVERY = 10" in nb_settings   # 60 秒 × 10 = 600 秒


def test_restart_republishes_the_url():
    """張り直したら `publish_url` まで呼ぶ（url.txt / Gist / ビーコン）。

    呼ばないと外は失効 URL を掴んだままになる。
    """
    src = _template()
    start = src.index("def restart_tunnel(procs):")
    end = src.index("def loop(procs", start)
    block = src[start:end]
    assert "publish_url(new_url)" in block
    assert "start_cloudflared(key=" in block, "再起動の計測が初回を上書きする"
    assert "procs[TUNNEL] = proc" in block, "新しいプロセスを差し替えていない"


@pytest.mark.parametrize("name", ["restart_tunnel", "_public_health",
                                  "save_evidence"])
def test_new_functions_are_in_the_script(name):
    """骨格に入っていること（builder 側の flow で落ちていないこと）。"""
    assert f"def {name}(" in _template()
    assert pathlib.Path(__file__).name == "test_serve_parity.py"
