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
