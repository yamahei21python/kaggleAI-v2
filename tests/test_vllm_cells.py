"""vLLM のセル（install / serve）の防御を固定する。

`tests/test_serve_script_guards.py` は**全エンジン共通の足場**（`serve_script.py` の
テンプレート）を持つ。こちらは vLLM 固有のセル（`kaggle/step2-vllm/cells/*.py`）で
実際に踏んだ不具合を、セル単位でまとめる。

**同じアサーションを 2 箇所に置かない**方針:
- 「症状 → 防御」の一覧（ユーザーの 13 項目）は `tests/test_gpu_pitfalls.py`
- 計測（timings / 節目の解析）は `tests/test_startup_timing.py`
- ここは「セルの順序・版の出どころ」など**セル内部の契約**だけ
"""
import glob as _glob  # noqa: E402

import pytest  # noqa: E402

import _cells  # noqa: E402

INSTALL = ("kaggle", "step2-vllm", "cells", "01_install_vllm.py")
SERVE = ("kaggle", "step2-vllm", "cells", "03_vllm_serve.py")


# ── install: 消してから入れる / 版は METADATA から ──────────────
def test_install_purges_torch_before_pip():
    """`pip uninstall torch` では RECORD に無い残骸を消せない。

    実測: 旧版の `_config_module.py` と新版の `torch/_dynamo/config.py` が同居し
    `TypeError: Config() got an unexpected keyword argument 'deprecated'`。
    物理削除が pip より**前**に走ることを順序で担保する。
    """
    src = _cells.cell(*INSTALL)
    assert "shutil.rmtree" in src, "物理削除が無い（pip uninstall では消し切れない）"
    assert src.index("_purge_torch()") < src.index('"-m", "pip", "install"')


def test_torchaudio_version_comes_from_vllm_metadata():
    """torchaudio の版は torch の版ではない（0.32 は torch 2.13 + audio 2.11）。

    torch に合わせて 2.13.0 を探すと存在せず、1 往復無駄にする。
    """
    src = _cells.cell(*INSTALL)
    assert '_vllm_pin("torchaudio")' in src
    assert "_torch_ver" not in src, "torch の版から torchaudio を決めている"


def test_installer_defaults_to_pip():
    """uv は**既定にしない**。CPU ベンチの比が本番に移らなかったため。

    実測 2026-11-01:
    - CPU ベンチ（`pip install --target`、まっさらな D）… uv 35.7s / pip 200.1s
    - GPU 本番（torch を消した site-packages へ）… pip 261.6s / **uv 1151.7s**

    同じ wheel・同じホストで 4.4 倍**遅い**。理由が未同定のまま既定にすると、
    1 回の実行で 20 分を捨てる（v12 がそうなった）。明示指定のときだけ試す。
    """
    src = _cells.cell(*INSTALL)
    assert 'KC_INSTALLER' in src, "uv を明示指定でしか使えないようになっていない"
    assert src.index('_installer = "pip"') < src.index('if _uv:'), (
        "既定が pip になっていない（uv が先に走る）")


# ── serve: 起動待ちの可観測性 ────────────────────────────────
def test_startup_wait_reports_last_milestone():
    """起動待ちの 60 秒ログが「末尾 1 行」だけだと、長い工程の中身が消える。

    実測 2026-10-31（#2 初成功）: torch.compile 終了 514s から CUDA graph capture
    開始 874s まで **360 秒**、ログは `shm_broadcast ... 60 seconds` の繰り返しだけで
    何をしていたか特定できなかった。節目を別枠で出す実装をここで固定する。
    """
    src = _cells.cell(*SERVE)
    assert "_MILESTONES" in src, "節目の一覧が無い"
    assert "節目:" in src, "起動待ちの出力に節目が出ていない"
    for need in ("Loading weights took", "torch.compile took",
                 "Capturing CUDA graphs"):
        assert need in src, (
            f"節目 {need!r} が無い。起動がどこで止まったか分からなくなる")


# ── serve: 重みロードの先読み ────────────────────────────────
def _prewarm_ns(tmp_path, files, read_range=None, sub="model"):
    """`_prewarm_weights` を tmp の model dir で動かせる状態にする。

    `read_range` を差し替えられるようにしてあるのは、**時間に依存する分岐**
    （並列が効くか / 予算に収まるか）を実 I/O なしで固定するため。
    """
    import os
    import time

    d = tmp_path / sub
    d.mkdir(exist_ok=True)
    for name, size in files.items():
        (d / name).write_bytes(b"x" * size)
    ns = {"MODEL_SRC": str(d), "os": os, "time": time}
    src = _cells.cell(*SERVE)
    for _name in ("_pool", "_read_range", "_prewarm_weights"):
        exec(_cells.slice_def(src, _name), ns)       # noqa: S102
    if read_range is not None:
        ns["_read_range"] = read_range
    return ns


def test_prewarm_reads_every_shard_and_reports_throughput(tmp_path):
    """先読みは**全部の shard** を読む。1 本でも漏らすと、そこだけ遅い経路に戻る。

    CPU 実測（`vllm-io-bench` 2026-11-01）: GCSFUSE の 1 本読みは 43.8 MiB/s、
    page cache に載った後の mmap は 5,402 MiB/s。v10 の重み 133 秒
    （140 MiB/s）は並列読みの値なので、同じ並列度で先に読む。
    """
    big = 3 * 1024 * 1024 + 17        # chunk で割り切れない端数も読む
    ns = _prewarm_ns(tmp_path, {"model.safetensors": big,
                                "model-mtp.safetensors": 1024 * 1024,
                                "config.json": 2})
    out = ns["_prewarm_weights"](threads=4, chunk_mib=1, budget_s=60)

    assert out["bytes"] == big + 1024 * 1024, "全 shard を読んでいない"
    assert out["aborted"] is False
    assert out["mib_s"] and out["mib_s"] > 0


def test_prewarm_skips_when_it_cannot_finish_in_budget(tmp_path):
    """予算に収まらない見込みなら**始めない**。

    GCSFUSE は並列が効かないことがある。効かないのに 8 本で読むと、保険の
    つもりが「133 秒で済むはずが 400 秒以上溶かす装置」になる。部分的な先読みは
    「残りが遅い」という症状を防げないので、時間だけ払って何も変わらない。
    """
    import time

    def slow_read(path, off, nbytes, threads=1, chunk_mib=32):
        time.sleep(0.3)           # 1 MiB / 0.3s ≒ 3.3 MiB/s
        return nbytes

    ns = _prewarm_ns(tmp_path, {"model.safetensors": 1024 * 1024},
                     read_range=slow_read)
    out = ns["_prewarm_weights"](threads=8, chunk_mib=1, budget_s=0.1)
    assert out.get("skipped") == "over budget", out
    assert out["projected_s"] > 0.1, "見込みを残していない（判定の根拠が消える）"
    assert "bytes" not in out, "始めてしまっている"


def test_prewarm_picks_the_faster_thread_count(tmp_path):
    """並列が効くなら N 本、効かないなら 1 本。**実測で選ぶ**（決め打ちにしない）。"""
    import time

    def scales(path, off, nbytes, threads=1, chunk_mib=32):
        time.sleep(0.02 if threads == 1 else 0.001)
        return nbytes

    ns = _prewarm_ns(tmp_path, {"model.safetensors": 2 * 1024 * 1024},
                     read_range=scales)
    out = ns["_prewarm_weights"](threads=8, chunk_mib=1, budget_s=60)
    assert out["threads"] == 8, out

    def worse(path, off, nbytes, threads=1, chunk_mib=32):
        time.sleep(0.02 if threads == 1 else 0.06)   # 並列の方が遅い
        return nbytes

    ns2 = _prewarm_ns(tmp_path, {"model.safetensors": 2 * 1024 * 1024},
                      read_range=worse, sub="model2")
    out2 = ns2["_prewarm_weights"](threads=8, chunk_mib=1, budget_s=60)
    assert out2["threads"] == 1, out2
    assert "bytes" in out2, "1 本に落としたうえで読むべき（並列が効かないだけ）"


def test_prewarm_stops_at_the_time_budget(tmp_path):
    """走り出したあとで速度が崩れたら予算で打ち切る。

    予算は起動試行の外側（`STARTUP_BUDGET` は先読みの後に起点を置き直す）だが、
    先読み自体が無制限だと 1 往復まるごと溶ける。上限は必ず要る。
    """
    ns = _prewarm_ns(tmp_path, {"model.safetensors": 1024 * 1024})
    out = ns["_prewarm_weights"](threads=2, chunk_mib=1, budget_s=0)
    # 予算 0 は「見込み > 0」なので**始めない**側に倒れる。
    assert out.get("skipped") == "over budget" or out.get("aborted") is True, out
    assert out.get("bytes", 0) == 0, "予算 0 なのに読んでいる"


def test_prewarm_defaults_to_off():
    """先読みの既定は **OFF**（2026-11-01 に反転）。

    先読みは「152.5 秒の純コスト」では済まない。後続の重みロードまで遅くする。
    - out2g（先読みなし）: 重みロード 約 181s → 起動 642.4s → 31.7 t/s
    - v14 （先読みあり）: 先読み 152.5s → 重みロード **705.00s** → 921s 打ち切り → 4.5 t/s

    同じマシンで 181s と 705s。先読み自身が測った 122.2 MiB/s なら 18.21 GiB は
    約 150s なので、181s は説明がつき 705s は説明がつかない（4.7 倍遅い）。
    既定に戻すと 1 回 20 分を捨てるので、ここで固定する。
    """
    src = _cells.cell(*SERVE)
    assert 'os.environ.get("KC_PREWARM", "0")' in src, (
        "先読みが既定 ON に戻っている（v14 で 1876.7s / 4.5 t/s になった構成）")


def test_prewarm_can_still_be_forced_on():
    """A/B 用の口は残す（既定を反転しても切り分けはできる）。

    `KC_PREWARM=1` で明示したときだけ走る。予算と並列度も外から変えられる。
    """
    src = _cells.cell(*SERVE)
    assert "KC_PREWARM" in src, "先読みを止められない（切り分けができない）"
    assert "KC_PREWARM_BUDGET" in src, "予算を外から変えられない"
    assert "KC_PREWARM_THREADS" in src, "並列度を外から変えられない"
    assert "先読みしない" in src, "効かないときに黙って読んでいる"


def test_ram_trace_is_saved_on_both_paths():
    """RAM の推移は**失敗した回にこそ**要る。成功時だけ保存すると意味が無い。

    実測 2026-11-01（v12）: 30,315 MiB → 5.68 GiB の 24 GiB がどこへ消えたか
    分からず、対策の順序を決められなかった。60 秒ポーリングでは 1 区間に潰れる。
    """
    src = _cells.cell(*SERVE)
    assert "def _ram_trace(" in src
    assert "threading.Thread" in src, "別スレッドで採っていない（待機で止まる）"
    # 保存はループの**外**（成功・失敗の合流点）に置く。
    after_loop = src.split("if not started:")[0]
    assert 'save_output("ram_trace.txt"' in after_loop, (
        "失敗経路で ram_trace が保存されない")


# ── Triton キャッシュ（起動の 54% を占める無音）────────────────
def _triton_ns(tmp_path, entries=(), dirs=None):
    """`_triton_*` を tmp のキャッシュディレクトリで動かせる状態にする。"""
    import os

    triton = tmp_path / ".triton"
    cache = triton / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in entries:
        (cache / name).mkdir()
    ns = {"os": os,
          "_TRITON_CACHE": str(cache),
          "_TRITON_DIRS": tuple(dirs) if dirs is not None else (str(triton),)}
    src = _cells.cell(*SERVE)
    for _name in ("_triton_snapshot", "_triton_report"):
        exec(_cells.slice_def(src, _name), ns)       # noqa: S102
    return ns


def test_triton_snapshot_happens_before_the_budget_starts():
    """「起動前の顔ぶれ」は**予算の起点より前**に控える。

    後に置くと、キャッシュの走査（数千エントリ）が試行の予算を食う。
    先読みを予算の外に出したのと同じ理由。
    """
    src = _cells.cell(*SERVE)
    # `_t_start` は 2 回出る（先読み前の仮置きと、先読み後の本番）。
    # 試行ループが使うのは**後者**なので `rindex` で比べる。
    assert src.index("_TRITON_BEFORE = _triton_snapshot()") < \
        src.rindex("_t_start = time.time()"), "予算の起点より後に控えている"


def test_triton_dirs_are_deduped_by_real_path(tmp_path):
    """同じディレクトリを 2 回 tar に入れない（実体で重複を落とす）。

    `~/.triton` と `/root/.triton` は HOME が `/root` のとき同一で、
    `dict.fromkeys` だけでは文字列が違うため落ちない。tar が二重になり、
    3 GB の上限に早く当たって「退避できたはずの回」を落とす。
    """
    import os

    real = tmp_path / "t"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real)
    ns = {"os": os}
    exec(_cells.slice_def(_cells.cell(*SERVE), "_uniq_dirs"), ns)

    got = ns["_uniq_dirs"]([str(real), str(link), str(real)])
    assert len(got) == 1, got


def test_triton_cache_dir_is_pinned_for_the_subprocess():
    """`TRITON_CACHE_DIR` を明示する。

    既定は Triton の版によって `~/.triton/cache` と `~/.cache/triton` の
    どちらにもなり得る。復元（01 の tar）と食い違うと、**復元したのに 1 件も
    当たらないまま 373 秒が戻る**。環境変数で 1 本に固定する。
    """
    src = _cells.cell(*SERVE)
    assert '"TRITON_CACHE_DIR": _TRITON_CACHE' in src, (
        "起動プロセスに TRITON_CACHE_DIR を渡していない")


def test_triton_measurement_counts_new_entries(tmp_path):
    """当たったかどうかを**エントリの増分**で判定する。

    「ファイルがある」では足りない。前回と別のカーネルを作った場合
    （キーの不一致＝ミス）も「ファイルはある」になる。増分なら区別できる。
    """
    ns = _triton_ns(tmp_path, entries=("aaa", "bbb", "ccc"))
    before = ns["_triton_snapshot"]()

    # 1 件だけ新規に作られた ＝ 3/4 が再利用された。
    (tmp_path / ".triton" / "cache" / "ddd").mkdir()
    rep = ns["_triton_report"](before, ns["_triton_snapshot"]())
    assert rep["entries_before"] == 3 and rep["entries_after"] == 4
    assert rep["new_entries"] == 1
    assert rep["reused_pct"] == 75.0

    # 全とっかえ ＝ キーが違って全部ミスした状態。ここを見落とすと
    # 「キャッシュは入れた」で終わってしまう。
    ns2 = _triton_ns(tmp_path / "b", entries=("xxx", "yyy"))
    before2 = ns2["_triton_snapshot"]()
    rep2 = ns2["_triton_report"](
        {"dir": before2["dir"], "names": {"old1", "old2"}}, before2)
    assert rep2["reused_pct"] == 0.0, rep2
    assert rep2["new_entries"] == 2


def test_triton_report_is_json_safe(tmp_path):
    """集合をそのまま timings に書かない（`json.dump` が落ちる）。"""
    import json

    ns = _triton_ns(tmp_path, entries=("aaa",))
    rep = ns["_triton_report"](ns["_triton_snapshot"](),
                               ns["_triton_snapshot"]())
    json.dumps(rep)          # 落ちたら集合が残っている
    assert "names" not in rep


def test_triton_snapshot_is_quiet_when_the_cache_is_absent(tmp_path):
    """キャッシュが無いのは**正常**（初回）。例外にしてはいけない。"""
    ns = _triton_ns(tmp_path, dirs=(str(tmp_path / "nope"),))
    ns["_TRITON_CACHE"] = str(tmp_path / "nope" / "cache")   # 実在しない
    snap = ns["_triton_snapshot"]()
    assert snap["names"] == set() and snap["dir"] == ""


def test_warm_cache_capture_writes_a_restorable_tar(tmp_path):
    """tar の arcname は復元側（`extractall("/")`）と一致していなければならない。

    ここがずれると「復元は成功したのに Triton から見えない」になり、
    サイズだけのログでは気づけない（実測: 01 の復元はサイズしか出していない）。
    """
    import os
    import tarfile

    ns = _triton_ns(tmp_path, entries=("k1", "k2"))
    (tmp_path / ".triton" / "cache" / "k1" / "a.cubin").write_bytes(b"x" * 100)
    ns.update({"OUT_DIRS": [str(tmp_path / "out")], "time": __import__("time")})
    exec(_cells.slice_def(_cells.cell(*SERVE), "_capture_warm_cache"), ns)

    out = ns["_capture_warm_cache"]()
    assert out.get("path"), out
    assert os.path.isfile(out["path"])
    with tarfile.open(out["path"]) as tf:
        names = tf.getnames()
    want = os.path.relpath(str(tmp_path / ".triton"), "/")
    assert any(n.startswith(want) for n in names), names
    assert any(n.endswith("k1/a.cubin") for n in names), names


def test_warm_cache_capture_never_fails_the_run(tmp_path):
    """退避に失敗しても起動は成功扱いのままにする。

    ここで例外を投げると「キャッシュが取れなかっただけで、起動できた回」を
    失う。373 秒を払った回を捨てる理由が無い。
    """
    import os

    ns = _triton_ns(tmp_path, entries=("k1",))
    # `out` を**ファイル**として置き、makedirs を失敗させる。
    (tmp_path / "out").write_text("not a dir")
    ns.update({"OUT_DIRS": [str(tmp_path / "out")], "time": __import__("time")})
    exec(_cells.slice_def(_cells.cell(*SERVE), "_capture_warm_cache"), ns)

    res = ns["_capture_warm_cache"]()          # 例外にならないこと
    assert "error" in res or res.get("skipped"), res
    assert not os.path.isfile(os.path.join(str(tmp_path / "out"),
                                          "warm-cache.tar"))


def test_warm_cache_capture_respects_the_size_limit(tmp_path):
    """3 GB を超えるキャッシュは持ち出さない（転送と容量の無駄）。"""
    import os

    ns = _triton_ns(tmp_path, entries=("k1",))
    (tmp_path / ".triton" / "cache" / "k1" / "a.cubin").write_bytes(b"x" * 4096)
    ns.update({"OUT_DIRS": [str(tmp_path / "out")], "time": __import__("time")})
    exec(_cells.slice_def(_cells.cell(*SERVE), "_capture_warm_cache"), ns)

    res = ns["_capture_warm_cache"](limit_bytes=1)
    assert res.get("skipped") == "too big", res
    assert not os.path.exists(os.path.join(str(tmp_path / "out"),
                                           "warm-cache.tar"))


def test_warm_cache_is_captured_on_both_paths():
    """退避は成功・失敗の**合流点**に置く。

    v14 は `engine_cleanup()`（キープアライブ終了時）が一度も走らず、
    `warm-cache.tar` が成果物に無かった。373 秒かけて作ったカーネルは
    失敗した回でもディスクにあるので、両方で持ち出す。
    """
    src = _cells.cell(*SERVE)
    after_loop = src.split("if not started:")[0]
    assert "_WARM_CACHE = _capture_warm_cache()" in after_loop, (
        "失敗経路でキャッシュが退避されない")



def _exec_warm_cache_block(tmp_path, mode):
    """install セルの warm cache ブロックだけを exec して、結果を返す。

    ブロックは `os` / `_glob` / `kc_find_dataset` にしか依存しないので、この 3 つを
    差し替えれば単体で動く。Dataset を**空のディレクトリ**に見せかけることで
    「Dataset はあるが tar が無い」＝実際に起きる最悪の形を再現する。
    """
    import os as _os

    src = _cells.cell(*INSTALL)
    start = src.index("# ---- warm cache（任意 Dataset）----")
    end = src.index("# ---- torch を「消してから入れる」ための道具 ----")
    block = src[start:end]

    empty = tmp_path / "empty-dataset"
    empty.mkdir(exist_ok=True)

    ns = {"os": _os, "_glob": _glob, "print": lambda *a, **k: None,
          "kc_find_dataset": lambda slug: str(empty)}
    old = _os.environ.get("KC_WARM_CACHE")
    _os.environ["KC_WARM_CACHE"] = mode
    try:
        exec(compile(block, "cell:01_install_vllm:warm_cache", "exec"), ns)
    finally:
        if old is None:
            _os.environ.pop("KC_WARM_CACHE", None)
        else:
            _os.environ["KC_WARM_CACHE"] = old


def test_warm_cache_require_raises_on_an_empty_dataset(tmp_path):
    """`require` は tar が 1 本も無ければ**実行時に**落ちる（テキストでなく挙動）。

    これを落とさないと、cache 実験が baseline と同じ数字を出し、
    「キャッシュは効かない」という偽陰性になる。
    """
    with pytest.raises(RuntimeError) as e:
        _exec_warm_cache_block(tmp_path, "require")
    assert "require" in str(e.value)
    assert "偽陰性" in str(e.value), "落ちる理由がメッセージに無い"


def test_warm_cache_default_mode_continues_on_an_empty_dataset(tmp_path):
    """既定 `1` は落ちない（本番が Dataset 未作成でも起動できること）。

    `require` を既定にすると、Dataset を作る前の本番が全部止まる。
    """
    _exec_warm_cache_block(tmp_path, "1")      # 例外が出なければよい


def test_warm_cache_off_never_searches_for_tars(tmp_path):
    """`0` は tar を探しにも行かない（対照群の条件を汚さない）。

    `kc_find_dataset` 自体は常に呼ばれる（パスを返すだけの安い処理）。
    契約は「**tar を探さない＝復元しない**」なので、`_glob` を見張る。
    """
    import os as _os

    src = _cells.cell(*INSTALL)
    start = src.index("# ---- warm cache（任意 Dataset）----")
    end = src.index("# ---- torch を「消してから入れる」ための道具 ----")
    block = src[start:end]

    globbed = []

    def spy_glob(pattern):
        globbed.append(pattern)
        return []

    ns = {"os": _os, "_glob": spy_glob, "print": lambda *a, **k: None,
          "kc_find_dataset": lambda slug: "/nonexistent"}
    old = _os.environ.get("KC_WARM_CACHE")
    _os.environ["KC_WARM_CACHE"] = "0"
    try:
        exec(compile(block, "cell:01_install_vllm:warm_cache", "exec"), ns)
    finally:
        if old is None:
            _os.environ.pop("KC_WARM_CACHE", None)
        else:
            _os.environ["KC_WARM_CACHE"] = old
    assert globbed == [], f"KC_WARM_CACHE=0 なのに tar を探した: {globbed}"
