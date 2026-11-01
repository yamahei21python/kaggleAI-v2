"""起動時間の計測（高度化の基礎）を固定する。

計測が壊れても実行は成功するので、**気づかないまま比較不能になる**。
ここで守るのは 3 点:

1. ログ時刻（1 秒精度）で節目を拾う（60 秒ポーリングでは 41 秒の区間が測れない）
2. 0 時跨ぎで負の値にならない
3. 取れなかった区間は `None`（0 で埋めると「一瞬で終わった」と誤読する）
"""
import _cells  # noqa: E402

SERVE_CELL = _cells.ROOT / "kaggle" / "step2-vllm" / "cells" / "03_vllm_serve.py"
INSTALL_CELL = _cells.ROOT / "kaggle" / "step2-vllm" / "cells" / "01_install_vllm.py"


def _ns():
    """`_phase_marks` / `_phase_spans` / `_long_gaps` と依存を読み込む。"""
    src = SERVE_CELL.read_text(encoding="utf-8")
    lines = src.splitlines()
    ms = next(i for i, ln in enumerate(lines) if ln.startswith("_MILESTONES = ("))
    me = ms
    while not lines[me].rstrip().endswith(")"):
        me += 1
    ns = {}
    exec("\n".join(lines[ms:me + 1]), ns)                      # noqa: S102
    exec("import re\n" + "\n".join(
        ln for ln in lines
        if ln.startswith("_TS_RE") or ln.startswith("_TS_ONLY")), ns)  # noqa: S102
    exec(_cells.slice_def(src, "_phase_marks"), ns)            # noqa: S102
    exec(_cells.slice_def(src, "_phase_spans"), ns)            # noqa: S102
    exec(_cells.slice_def(src, "_long_gaps"), ns)              # noqa: S102
    return ns


# 実際のログ行に近い形（`10-31 13:22:08` が行中に入る）。
_LOG = """\
(APIServer pid=122) INFO 10-31 13:21:50 [api_server.py:10] Starting vLLM API server
(Worker_TP0 pid=293) INFO 10-31 13:21:52 [loader.py:10] Starting to load model
(Worker_TP0 pid=293) INFO 10-31 13:23:55 [monitor.py:40] [backbone] Starting torch.compile
(Worker_TP0 pid=293) INFO 10-31 13:24:56 [monitor.py:62] [backbone] torch.compile took 61.22 s in total
(EngineCore pid=259) INFO 10-31 13:29:53 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds.
"""


def test_phase_marks_use_the_log_clock():
    """ログの時刻から起動からの秒を出す（60 秒ポーリングでは精度が足りない）。"""
    ns = _ns()
    t0 = 13 * 3600 + 21 * 60 + 48          # 13:21:48 に起動
    marks = ns["_phase_marks"](_LOG, t0)
    assert marks["Starting to load model"] == 4.0
    assert marks["Starting torch.compile"] == 127.0
    assert marks["torch.compile took"] == 188.0
    # 時刻の無い行（shm_broadcast は時刻あり、tqdm は無し）は拾わない
    assert "Capturing CUDA graphs" not in marks


def test_phase_marks_survive_midnight_wrap():
    """0 時をまたいでも負にならない（23:59:50 起動 → 00:00:10 の節目）。"""
    ns = _ns()
    log = "(W pid=1) INFO 11-01 00:00:10 [x.py:1] Starting torch.compile\n"
    t0 = 23 * 3600 + 59 * 60 + 50          # 86390
    assert ns["_phase_marks"](log, t0)["Starting torch.compile"] == 20.0


def test_phase_marks_keep_the_first_occurrence():
    """同じ節目が複数回出たら最初を採る（最後だと区間が縮んで見える）。"""
    ns = _ns()
    log = (
        "(W pid=1) INFO 10-31 13:00:10 [x.py:1] Starting torch.compile\n"
        "(W pid=2) INFO 10-31 13:05:00 [x.py:1] Starting torch.compile\n"
    )
    t0 = 13 * 3600
    assert ns["_phase_marks"](log, t0)["Starting torch.compile"] == 10.0


def test_phase_spans_distinguish_missing_from_zero():
    """片方の節目が無ければ None。両方あって同時刻なら 0。"""
    ns = _ns()
    spans = ns["_phase_spans"]({
        "Starting torch.compile": 127.0,
        "torch.compile took": 188.0,
    })
    assert spans["compile"] == 61.0
    assert spans["weights"] is None, "取れていない区間を 0 で埋めている"
    assert spans["capture"] is None

    zero = ns["_phase_spans"]({"Starting torch.compile": 10.0,
                               "torch.compile took": 10.0})
    assert zero["compile"] == 0.0, "同時刻は 0（None ではない）"


def test_install_cell_records_install_seconds():
    """`timings.install_seconds` が None のままだった（比較不能の原因）。"""
    src = INSTALL_CELL.read_text(encoding="utf-8")
    assert "kc_save_timings(" in src, "install の所要を timings.json に入れていない"
    assert '"install_seconds"' in src
    assert '"install_marks"' in src, "内訳（pip/torchaudio/help）を残していない"
    assert "_mark(" in src


def test_serve_cell_records_load_seconds_and_marks():
    """起動の所要と内訳を result.json に残す（毎回の比較の土台）。"""
    src = SERVE_CELL.read_text(encoding="utf-8")
    assert '"load_seconds"' in src, "起動の所要を入れていない"
    assert '"startup_marks"' in src, "節目を入れていない"
    assert '"startup_spans"' in src, "区間を入れていない"
    assert "_poll_marks" in src, (
        "graph capture（tqdm・時刻なし）を拾うポーリング側の控えが無い")
    assert "{**_poll_marks, **_marks}" in src, (
        "ログ時刻（1 秒）を優先していない、またはポーリングで補完していない")


def test_cells_do_not_touch_the_script_only_data_dict():
    """セルが `data` を直接触ると **notebook 版が壊れる**（実測 2026-11-01）。

    notebook には `data` / `save()` が存在しない（テンプレート専用）。
    セル順の「定義前使用」テストが落ちるだけでなく、実行時も NameError になる。
    計測は timings.json 経由にして、両方の形で同じセルが動くようにする。
    """
    for path in (INSTALL_CELL, SERVE_CELL):
        src = path.read_text(encoding="utf-8")
        for ln in src.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                continue
            assert not s.startswith('data["'), (
                f"{path.name}: セルが data を直接書いている → {s}")
            assert s != "save()", f"{path.name}: セルが save() を呼んでいる"


def test_template_merges_cell_timings_into_result():
    """timings.json を result.json に合流させないと、計測が成果物に残らない。"""
    from kbuild import serve_script
    # 切り出し元はテンプレート本体（ファイル全体だと関数は文字列の内側）。
    tpl = serve_script.TEMPLATE.template
    assert 'TIMINGS = "/kaggle/working/timings.json"' in tpl
    assert "data[\"timings\"]" in tpl
    # 切り出しは `_cells` に任せる（`def save():` の文字列分割だと、同名の
    # 定義が増えたときに黙って別のものを掴む）。
    assert "TIMINGS" in _cells.slice_def(tpl, "save")



# ── 無音区間（実測 2026-11-01 の形をそのまま使う）──────────────
# 起動からの秒に直すと 282（compile 終了）→ 339/399/459/519/579（60 秒警告）
# → 663（graph capture 終了）。合計 381 秒が「何も出ない区間」。
_GAP_LOG = """\
(Worker_TP1 pid=294) INFO 11-01 00:04:42 [monitor.py:62] [backbone] torch.compile took 61.00 s in total
(EngineCore pid=260) INFO 11-01 00:05:39 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes
(EngineCore pid=260) INFO 11-01 00:06:39 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes
(EngineCore pid=260) INFO 11-01 00:07:39 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes
(EngineCore pid=260) INFO 11-01 00:08:39 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes
(EngineCore pid=260) INFO 11-01 00:09:39 [shm_broadcast.py:871] No available shared memory broadcast block found in 60 seconds. This typically happens when some processes
(EngineCore pid=260) INFO 11-01 00:11:03 [core.py:800] Graph capturing finished
"""


def test_long_gaps_merge_repeated_warnings_into_one_span():
    """60 秒警告の繰り返しを 1 本に連結する（5 本並ぶと「300 秒詰まり」に見えない）。"""
    ns = _ns()
    gaps = ns["_long_gaps"](_GAP_LOG, 0)
    assert len(gaps) == 1, gaps
    g = gaps[0]
    assert g["from"] == 282.0
    assert g["seconds"] == 381.0, "282→663 の 381 秒になっていない"
    assert "torch.compile took" in g["before"]
    assert "Graph capturing finished" in g["after"]


def test_long_gaps_name_the_lines_around_the_silence():
    """無音の**前後**を残す。長さだけでは次の一手が決まらない。"""
    ns = _ns()
    log = (
        "(W pid=1) INFO 11-01 00:00:10 [a.py:1] Loading weights took 133.00 s\n"
        "(W pid=1) INFO 11-01 00:05:10 [a.py:1] Capturing CUDA graphs\n"
    )
    gaps = ns["_long_gaps"](log, 0)
    assert len(gaps) == 1
    assert gaps[0]["seconds"] == 300.0
    assert "Loading weights took" in gaps[0]["before"]
    assert "Capturing CUDA graphs" in gaps[0]["after"]


def test_long_gaps_ignore_short_intervals():
    """30 秒未満は無音と呼ばない（通常のログ間隔まで拾うと全部が無音になる）。"""
    ns = _ns()
    log = (
        "(W pid=1) INFO 11-01 00:00:00 [a.py:1] step 1\n"
        "(W pid=1) INFO 11-01 00:00:05 [a.py:1] step 2\n"
        "(W pid=1) INFO 11-01 00:00:09 [a.py:1] step 3\n"
    )
    assert ns["_long_gaps"](log, 0) == []


def test_long_gaps_survive_midnight_wrap():
    """0 時跨ぎで負の区間にならない（無音が消えると原因追跡ができなくなる）。"""
    ns = _ns()
    log = (
        "(W pid=1) INFO 10-31 23:59:00 [a.py:1] before\n"
        "(W pid=1) INFO 11-01 00:01:00 [a.py:1] after\n"
    )
    t0 = 23 * 3600 + 58 * 60            # 23:58:00 に起動
    gaps = ns["_long_gaps"](log, t0)
    assert len(gaps) == 1
    assert gaps[0]["seconds"] == 120.0


def test_successful_attempt_log_is_saved():
    """**成功時こそ**ログを残す（失敗時だけ保存していたため原因を読めなかった）。

    実測 2026-11-01: 起動は成功したが `vllm_1.log` が成果物に無く、
    379 秒の無音区間の中身を確認できなかった。
    """
    src = SERVE_CELL.read_text(encoding="utf-8")
    head = src.split("if not started:")[0]
    assert 'save_output(f"vllm_{idx}.log"' in head, (
        "成功経路でログを保存していない（break が保存行を飛ばす）")
    assert '"startup_gaps"' in head, "無音区間を result.json に残していない"


def test_warm_cache_reports_missing_dataset(tmp_path):
    """`kc_find_dataset` は未マウントでもパスを返す → `if _wc:` は常に真。

    そのため tar が 1 本も無いときに**何も表示せず素通り**していた
    （実測 2026-11-01: ログに warm cache の行が 1 行も無い）。
    """
    src = INSTALL_CELL.read_text(encoding="utf-8")
    assert "_tars = " in src, "tar の有無を判定していない"
    assert "exists=" in src, "未マウントと tar 欠落を切り分けられない"


def test_install_cell_prefers_uv_but_keeps_pip_as_fallback():
    """install は uv を優先し、失敗したら pip に戻る（実測 200.1s → 35.7s）。

    CPU ベンチ `vllm-install-bench`（2026-11-01）の結果。差の大半は pyc の生成
    （25,428 個）と pip の直列展開で、**転送ではない**（wheelhouse を付けても
    pip は 199.6 → 230.4s と遅くなった）。
    uv は rc=0 でも部分的にしか入らないことがあるので、`vllm` CLI の実在まで
    確かめる。無ければ pip で入れ直す（起動できない方が損失が大きい）。
    """
    src = INSTALL_CELL.read_text(encoding="utf-8")
    assert 'shutil.which("uv")' in src, "uv を探していない"
    assert '"--python", sys.executable' in src, "uv に python を渡していない"
    assert 'shutil.which("vllm")' in src, "uv の部分成功を検出できない"
    assert '_installer = "uv"' in src and '_installer = "pip"' in src
    assert '"installer": _installer' in src, "どちらで入れたか残していない"


def test_install_bench_covers_four_strategies():
    """CPU ベンチが 4 通りを測る（測っていない戦略を「速い」と言わない）。

    実測値: pip 200.1s / pip --no-compile 112.6s / uv 35.7s /
    pip --no-deps+パス直指定 101.5s。baseline を最後にもう一度測って
    キャッシュ暖まりの影響（`cache_warm_bias`）を検出する。
    """
    bench = (_cells.ROOT / "kaggle" / "cpu-tests" / "install-bench" /
             "main.py").read_text(encoding="utf-8")
    for name in ("pip", "pip-nocompile", "uv", "pip-paths-nodeps", "pip-again"):
        assert f'"{name}"' in bench, f"{name} を測っていない"
    assert "cache_warm_bias" in bench, "キャッシュ暖まりの対照が無い"
    assert "gpu_unknown" in bench, "CPU で確定できない項目を明示していない"


def test_summarize_reads_nested_timings():
    """result.json の timings から 1 行分を取り出す。"""
    from cli.timings import summarize

    row = summarize({
        "engine": "vllm", "elapsed_seconds": 928.7, "ok": True,
        "context_length": 65536,
        "timings": {
            "install_seconds": 242.7, "load_seconds": 642.4,
            "speed_tps": 31.7, "gpu_vram_mib": [13343, 13343],
            "attempt": {"label": "mtp+dd", "max_model_len": 65536},
            "startup_spans": {"weights": 181.0, "compile": 61.2,
                              "after_compile": 360.0, "capture": 41.0},
        },
    })
    assert row["install"] == 242.7
    assert row["load"] == 642.4
    # 総計（928.7s）は keepalive を含むので比較に使えない。待たされる時間は
    # install + 起動。実測 2026-11-01 の総計 4637s のうち 3660s は配信だった。
    assert row["ready"] == 885.1
    assert row["speed"] == 31.7
    assert row["label"] == "mtp+dd"
    assert row["after_compile"] == 360.0


def test_summarize_reports_the_serving_torch_not_the_image_one():
    """`torch` は **install 後**の値を使う（result.json の `torch` はイメージ版）。

    実測 2026-11-01: result.json は `2.11.0+cu128`（install 前）だが、実際に
    動いたのは `2.13.0+cu130`。表だけ見ると「2.11 で動いた」と誤読する。
    """
    from cli.timings import summarize

    row = summarize({"engine": "vllm", "torch": "2.11.0+cu128",
                     "timings": {"torch_after_install": "2.13.0+cu130"}})
    assert row["torch"] == "2.13.0+cu130"
    # 記録が無い回は result.json の値にフォールバックする（列を空にしない）。
    assert summarize({"engine": "vllm", "torch": "2.11.0+cu128"})["torch"] == \
        "2.11.0+cu128"


def test_table_uses_dash_for_unmeasured_values():
    """測っていない値を 0 で埋めない（「一瞬で終わった」と誤読する）。"""
    from cli.timings import table

    out = table([{"engine": "vllm", "elapsed": 100, "install": None,
                  "load": None, "label": None, "maxlen": None, "speed": None,
                  "vram": None, "ok": True, "weights": None, "compile": None,
                  "after_compile": None, "capture": None}])
    body = out.splitlines()[-1]
    assert "-" in body, body
    assert "0.0" not in body, f"未計測が 0.0 になっている: {body}"


def test_timings_is_a_cli_subcommand():
    """計測を比べる導線が CLI にあること（無いと誰も使わない）。"""
    from cli.__main__ import _parser

    args = _parser().parse_args(["timings", "a.json", "b.json"])
    assert args.cmd == "timings"
    assert args.paths == ["a.json", "b.json"]


def test_save_timings_merges_two_cells_into_one_file(tmp_path):
    """install セルと serve セルが別々に書いても 1 つの JSON に合流する。

    上書きにすると、後から書いた方だけが残って install の計測が消える
    （＝ 比較の土台が片肺になる）。静かに壊れるので契約として固定する。
    """
    import json

    src = (_cells.ROOT / "kaggle" / "step2-vllm" / "cells" / "00_settings.py").read_text(
        encoding="utf-8")
    path = tmp_path / "timings.json"
    ns = {"json": json, "TIMINGS_PATH": str(path), "print": print}
    exec(_cells.slice_def(src, "kc_save_timings"), ns)               # noqa: S102
    save = ns["kc_save_timings"]

    save({"install_seconds": 242.7})
    save({"load_seconds": 642.4})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "install_seconds": 242.7, "load_seconds": 642.4}
