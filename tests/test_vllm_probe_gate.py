"""ノード速度ゲート（`KC_PROBE_GATE_MIB_S`）の契約。

`cli race` は「probe が閾値未満なら install に入る前に降りる」ことで成立する。
ここが緩むと、遅いノードで install 267 秒 + 重みロードを払ってから諦める
（実測 v18: 全体 10,576 秒）。
"""
import _cells  # noqa: E402

SETTINGS = ("kaggle", "step2-vllm", "cells", "00_settings.py")


def _gate():
    return _cells.exec_def(_cells.cell(*SETTINGS), "_kc_probe_gate")


def test_gate_off_always_passes():
    """0 / 未設定なら無効。従来どおり全部走る（既定を壊さない）。"""
    gate = _gate()
    rec = {"ok": True, "mib_s": 1.0}
    assert gate(rec, 0) is True
    assert gate(rec, -1) is True


def test_gate_rejects_below_threshold():
    """閾値未満は落とす（install 前に降りる）。"""
    gate = _gate()
    assert gate({"ok": True, "mib_s": 149.9}, 150.0) is False


def test_gate_accepts_equal_threshold():
    """境界は含む（150 ちょうどは通す）。比較を `>` にすると 1 件落ちる。"""
    gate = _gate()
    assert gate({"ok": True, "mib_s": 150.0}, 150.0) is True


def test_gate_passes_when_probe_failed():
    """probe が失敗した回は判定できないので通す（安全側 = 走らせる）。"""
    gate = _gate()
    assert gate({"ok": False, "why": "OSError"}, 150.0) is True
    assert gate({}, 150.0) is True


def test_gate_runs_before_install():
    """ゲートは `stage_install()` の**呼び出しより前**（モジュール読み込み時）。

    後ろに置くと install 267 秒を払ってから降りることになり、レースの
    1 ラウンドが 5 分になる。生成物の行位置で担保する。
    """
    main = _cells.cell("kaggle", "vllm-playground", "main.py")
    gate_at = main.index("KC_PROBE_GATE_MIB_S")
    install_at = main.index("def stage_install")
    assert gate_at < install_at


# ── beacon（走行中に probe 結果を外へ出す）──────────────────────
def _beacon(monkeypatch, sent, env):
    """`_kc_probe_beacon` を、urlopen を差し替えて返す。"""
    import os
    import urllib.request

    class _Resp:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        sent.append((req.full_url, req.data))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    for k in ("KC_PROBE_NTFY_TOPIC", "KC_PROBE_SIDE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    fn = _cells.exec_def(_cells.cell(*SETTINGS), "_kc_probe_beacon", os=os)
    return fn


def test_beacon_off_without_topic(monkeypatch):
    sent = []
    fn = _beacon(monkeypatch, sent, {})
    fn({"ok": True, "mib_s": 21.5})
    assert sent == []


def test_beacon_posts_mib_and_side(monkeypatch):
    sent = []
    fn = _beacon(monkeypatch, sent,
                 {"KC_PROBE_NTFY_TOPIC": "t-1", "KC_PROBE_SIDE": "a"})
    fn({"ok": True, "mib_s": 21.5})
    assert len(sent) == 1
    url, body = sent[0]
    assert url == "https://ntfy.sh/t-1"
    assert b"mib_s=21.5" in body and b"side=a" in body


def test_beacon_silent_on_probe_failure(monkeypatch):
    sent = []
    fn = _beacon(monkeypatch, sent, {"KC_PROBE_NTFY_TOPIC": "t-1"})
    fn({"ok": False, "why": "OSError"})
    assert sent == []
