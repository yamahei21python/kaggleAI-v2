#!/usr/bin/env python3
"""pinned な vLLM wheel の中身に依存している前提を、GPU を使う前に確かめる。

`playground` の `force_first` 実験は `VLLM_TRITON_FORCE_FIRST_CONFIG` を立てる。
この環境変数は **vLLM 側の実装**なので、vLLM を上げたときに消えていれば
**何も起きないまま 40 分の実験が終わる**（＝もう 1 つの偽陰性）。

`.wheel/` は 1.1 GB あり gitignore しているので、無い環境では skip する。
「CI で必ず走るテスト」ではなく「GPU を撃つ直前に効くテスト」として置く。
"""
import pathlib
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
WHEEL_GLOB = "vllm-*.whl"


def _wheel():
    found = sorted((ROOT / ".wheel").glob(WHEEL_GLOB))
    if not found:
        pytest.skip(f".wheel/{WHEEL_GLOB} が無い（1.1 GB のローカル資産。gitignore 済み）")
    return found[0]


def test_force_first_config_is_present_in_the_pinned_wheel():
    """`force_first` 実験が効く土台（2 ファイル）が wheel に入っていること。

    無いと `VLLM_TRITON_FORCE_FIRST_CONFIG=1` は単なる未知の環境変数になり、
    autotune はそのまま走る。起動時間が変わらないので「autotune ではなかった」と
    誤読する。
    """
    with zipfile.ZipFile(_wheel()) as z:
        names = set(z.namelist())
        assert "vllm/env_override.py" in names, "env_override.py が無い"
        assert "vllm/triton_utils/force_first_config.py" in names, (
            "force_first_config.py が無い。vLLM を上げたならフラグの行き先を確認する")

        hook = z.read("vllm/env_override.py").decode("utf-8", "replace")
        assert "VLLM_TRITON_FORCE_FIRST_CONFIG" in hook, (
            "env_override.py がフラグを読んでいない（名前が変わった可能性）")


def test_force_first_config_skips_benchmarking():
    """差し替え版が **ベンチしない** こと（＝起動時間が縮む余地があること）。

    `do_bench` を呼ぶ実装だと「全 config を計測してから 1 個選ぶ」だけになり、
    コンパイル時間は減らない。それでは 373 秒の判定に使えない。
    """
    with zipfile.ZipFile(_wheel()) as z:
        src = z.read("vllm/triton_utils/force_first_config.py").decode("utf-8", "replace")
    assert "do_bench" not in src, (
        "差し替え版がベンチしている。これではコンパイル回数が減らない")
    assert "Autotuner.run = " in src, "Autotuner.run を差し替えていない"


def test_force_first_config_logs_how_many_configs_existed():
    """`configs=` をログに出すこと。

    `force_first` の回にこの行を数えると「autotune が候補を何個持っていたか」が
    分かり、キャッシュで潰せる量の見積りになる。ログ形式が変わると数えられなくなる。
    """
    with zipfile.ZipFile(_wheel()) as z:
        src = z.read("vllm/triton_utils/force_first_config.py").decode("utf-8", "replace")
    assert "triton-autotune-disabled" in src, "数えるための目印がログに無い"
    assert "configs=" in src, "configs= が出ないと候補数が分からない"
