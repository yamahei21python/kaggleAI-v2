"""レースの判定表（`cli/race.py` の純関数）。

判定を間違えると「遅いノードを本番にして、速い方を delete」する。
ネットワークを触らない関数だけを固定する。
"""
from cli import race


def test_both_below_gate_is_retry():
    assert race.decide(20.0, 149.9, 150.0) == "retry"


def test_only_a_passes():
    assert race.decide(203.9, 25.0, 150.0) == "a"


def test_only_b_passes():
    assert race.decide(25.0, 180.2, 150.0) == "b"


def test_both_pass_keeps_faster():
    """両方通過なら probe が速い方だけ残す（両方走らせない）。"""
    assert race.decide(203.9, 180.2, 150.0) == "a"
    assert race.decide(180.2, 203.9, 150.0) == "b"


def test_both_pass_tie_keeps_a():
    """同値は a（決定的に。乱数や順序依存にしない）。"""
    assert race.decide(150.0, 150.0, 150.0) == "a"


def test_missing_probe_counts_as_failure():
    """probe が取れない = ゲート未満として扱う（None を数値比較しない）。"""
    assert race.decide(None, None, 150.0) == "retry"
    assert race.decide(None, 180.0, 150.0) == "b"
    assert race.decide(180.0, None, 150.0) == "a"


def test_parse_push_output():
    out = "Kernel version 12 successfully pushed.  Please check progress at ..."
    assert race.parse_version(out) == 12
    assert race.parse_version("") == 0


def test_parse_status_line():
    line = 'yamahei21python/x has status "KernelWorkerStatus.RUNNING"'
    assert race.parse_status(line) == "RUNNING"
    assert race.parse_status("") == ""


def test_stage_meta_title_resolves_to_id():
    """title が id に解決できないと push が 409 で落ちる（実測）。

    `title="race a"` は `yamahei21python/race-a` に化けて、指定 id
    `vllm-race-a` と衝突した。title は slug と同一にする。
    """
    meta = race.stage_meta({"dataset_sources": ["x/y"]}, "a", "owner")
    assert meta["id"] == "owner/vllm-race-a"
    assert meta["title"] == "vllm-race-a"
    assert meta["dataset_sources"] == ["x/y"], "元のメタデータを落としている"
