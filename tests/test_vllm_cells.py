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
import pathlib  # noqa: E402
import re  # noqa: E402

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


def test_prewarm_does_not_run_unconditionally():
    """先読みは**無条件には走らない**（2026-11-01 に既定 OFF）。

    先読みは「152.5 秒の純コスト」では済まない。後続の重みロードまで遅くする。
    - out2g（先読みなし）: 重みロード 約 181s → 起動 642.4s → 31.7 t/s
    - v14 （先読みあり）: 先読み 152.5s → 重みロード **705.00s** → 921s 打ち切り → 4.5 t/s

    同じマシンで 181s と 705s。先読み自身が測った 122.2 MiB/s なら 18.21 GiB は
    約 150s なので、181s は説明がつき 705s は説明がつかない（4.7 倍遅い）。

    2026-11-02 に「vLLM が prefetch を辞退する見込みの回だけ自動」へ狭めた
    （v17 が辞退されたまま 6,128 秒を焼いたため）。prefetch が通るノードでは
    v14 と同じ構成＝先読みなしのまま。`KC_PREWARM=0` の明示で自動も止まる。
    """
    src = _cells.cell(*SERVE)
    assert 'os.environ.get("KC_PREWARM", "0")' not in src, (
        "無条件の既定 ON に戻っている（v14 で 1876.7s / 4.5 t/s になった構成）")
    assert 'if _WEIGHTS_GIB and _pw_want:' in src, "条件で囲まれていない"


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



# v15 が残した不完全な tar に入っていた 4 家族。**373 秒の主役が欠けている**。
_INCOMPLETE_KERNELS = ("_fused_post_conv", "_prepare_rope_positions",
                       "chunk_local_cumsum_scalar", "chunk_scaled_dot_kkt_fwd")
# 2026-11-01 の v4 が残した**完全な tar に実在した名前**（推測ではなく実測）。
# tar を数えた内訳: 2,077 エントリ / 195 MB / カーネル名 140 種、
# `chunk_fwd_kernel_o` x252、`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` x84。
_MEASURED_TAR_KERNELS = ("chunk_fwd_kernel_o",
                         "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
_COMPLETE_KERNELS = _INCOMPLETE_KERNELS + _MEASURED_TAR_KERNELS


def _exec_warm_cache_block(tmp_path, mode, kernels=None, tree=False):
    """install セルの warm cache ブロックだけを exec して `_WC_INFO` を返す。

    ブロックは `os` / `_glob` / `kc_find_dataset` にしか依存しないので、この 3 つを
    差し替えれば単体で動く。Dataset を**空のディレクトリ**に見せかけることで
    「Dataset はあるが tar が無い」＝実際に起きる最悪の形を再現する。

    `kernels` を渡すと、そのカーネルだけが入った tar を展開した状態を作る。
    展開だけ差し替えるのは、ブロックが `extractall("/")` を呼ぶため——tar の
    arcname 次第で**どこへでも書く**ので、テストで本物を走らせてはいけない。
    数える側（`os.walk`）は本物を通すので、「名前を見ている」こと自体は検証できる。

    `tree=True` にすると、tar ではなく**展開済みの木**（`root/.triton/cache/...`）
    を Dataset に置く。Kaggle は上げた tar を展開して保持するので、こちらが
    実際のマウント形（実測 2026-11-01）。コピーは本物を通す。
    """
    import os as _os
    import sys as _sys
    import types as _types

    src = _cells.cell(*INSTALL)
    start = src.index("# ---- warm cache（任意 Dataset）----")
    end = src.index("# ---- torch を「消してから入れる」ための道具 ----")
    block = src[start:end]

    ds = tmp_path / "dataset"
    ds.mkdir(exist_ok=True)
    if kernels is not None and tree:
        for i, name in enumerate(kernels):
            d = ds / "root" / ".triton" / "cache" / f"{i:04d}"
            d.mkdir(parents=True, exist_ok=True)
            for ext in ("json", "ptx", "cubin"):
                (d / f"{name}.{ext}").write_text("x")
    elif kernels is not None:
        (ds / "warm-cache.tar").write_bytes(b"\0" * 1024)   # getsize を呼ぶので実体が要る
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)

    class _FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extractall(self, root):
            # Triton の実配置に合わせる: 1 カーネル 1 ディレクトリで、
            # 中に `<カーネル名>.<ext>` が入る。
            tc = _os.path.join(_os.environ["HOME"], ".triton", "cache")
            for i, name in enumerate(kernels):
                d = _os.path.join(tc, f"{i:04d}")
                _os.makedirs(d, exist_ok=True)
                for ext in ("json", "ptx", "cubin"):
                    (pathlib.Path(d) / f"{name}.{ext}").write_text("x")

    fake = _types.ModuleType("tarfile")
    fake.open = lambda path: _FakeTar()

    ns = {"os": _os, "_glob": _glob, "print": lambda *a, **k: None,
          "kc_find_dataset": lambda slug: str(ds)}
    saved = {k: _os.environ.get(k) for k in ("KC_WARM_CACHE", "HOME")}
    orig_tarfile = _sys.modules.get("tarfile")
    _os.environ["KC_WARM_CACHE"] = mode
    _os.environ["HOME"] = str(home)
    _sys.modules["tarfile"] = fake
    try:
        exec(compile(block, "cell:01_install_vllm:warm_cache", "exec"), ns)
        return ns.get("_WC_INFO", {})
    finally:
        for k, v in saved.items():
            if v is None:
                _os.environ.pop(k, None)
            else:
                _os.environ[k] = v
        if orig_tarfile is not None:
            _sys.modules["tarfile"] = orig_tarfile
        else:
            _sys.modules.pop("tarfile", None)


def test_attempt_timeout_covers_the_whole_startup_budget():
    """1 試行の上限が総予算より小さいと、**遅い回は必ず切られる**。

    2026-11-01 の `baseline` は 2 試行とも「壊れた」のではなく予算で切られた:
    attempt 1 は 900 秒（`ATTEMPT_TIMEOUT`）で重みロードの途中、
    attempt 2 は 806 秒（残予算でクランプ）で SIGTERM。
    同じ回の vLLM ログは `Checkpoint size: 17.41 GiB / Available RAM: 5.85 GiB`
    で prefetch を切り、GCSFUSE からの読みが 20 MiB/s 前後まで落ちていた。
    17.41 GiB ÷ 20 MiB/s ≒ 900 秒に、profile run の 373 秒が乗る。

    「遅い」失敗と「非対応」失敗は別物。非対応は即座に落ちるので予算を食わず、
    次の段に回せる。遅いだけの試行を短い枠で切ると次段も同じ理由で落ちる。
    """
    src = _cells.cell(*SERVE)
    budget = re.search(
        r'STARTUP_BUDGET = int\(os\.environ\.get\("VLLM_STARTUP_BUDGET", "(\d+)"\)\)', src)
    attempt = re.search(
        r'ATTEMPT_TIMEOUT = int\(os\.environ\.get\("VLLM_ATTEMPT_TIMEOUT", "(\d+)"\)\)', src)
    assert budget and attempt, "既定値の定義が見つからない"
    b, a = int(budget.group(1)), int(attempt.group(1))
    assert a >= b, (
        f"1 試行の上限 {a}s が総予算 {b}s より小さい。遅い回は必ず切られる"
        "（実測 2026-11-01: 重みだけで 900 秒 + profile run 373 秒）")
    # 下限は**実測の遅い側**で決める。2026-11-01 の v3 は 2700 でも足りず、
    # 全 62 層を読み終えた t=2890s のあと t=2972s で打ち切られた
    # （重み 2,640s + profile 373s = 3,013s が下限）。2400 に戻すと
    # 「最後まで読んでから切る」を再現するだけになる。
    assert b >= 3600, (
        f"総予算 {b}s は実測の遅い側に足りない。重み 2,640s + profile 373s "
        "= 3,013s に余裕を持たせること")


def test_warm_cache_require_raises_on_an_empty_dataset(tmp_path):
    """`require` は tar が 1 本も無ければ**実行時に**落ちる（テキストでなく挙動）。

    これを落とさないと、cache 実験が baseline と同じ数字を出し、
    「キャッシュは効かない」という偽陰性になる。
    """
    with pytest.raises(RuntimeError) as e:
        _exec_warm_cache_block(tmp_path, "require")
    assert "require" in str(e.value)
    assert "偽陰性" in str(e.value), "落ちる理由がメッセージに無い"


def test_warm_cache_require_rejects_a_tar_without_the_profile_run_kernels(tmp_path):
    """復元はできるが**373 秒の主役が欠けている** tar も `require` では落とす。

    実測 2026-11-01: 重みロード中に落ちた回の tar は 30 エントリ / 70 KB で、
    `chunk_fwd_kernel_o` / `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` が
    入っていなかった（profile run に未到達）。復元すると一部だけ当たって
    「少し効いた」ように見えるので、件数ではなく**名前**で判定する
    （件数は Triton の版で変わり、しきい値を置くと正当な tar を弾くほうに倒れる）。
    """
    with pytest.raises(RuntimeError) as e:
        _exec_warm_cache_block(tmp_path, "require",
                               kernels=_INCOMPLETE_KERNELS)
    msg = str(e.value)
    assert "chunk_fwd_kernel_o" in msg and "chunk_gated_delta_rule_fwd_kernel_h" in msg, \
        f"何が欠けているか書いていない: {msg}"
    assert "偽陰性" in msg, f"落ちる理由がメッセージに無い: {msg}"


def test_warm_cache_guard_uses_the_measured_kernel_names():
    """ガードの条件は**実測のファイル名に一致**していること。

    最初 `chunk_o` / `chunk_delta_h` で書いたが、実際の名前は
    `chunk_fwd_kernel_o` / `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` で
    部分一致しない。そのまま `require` で走らせると、**このリポジトリが
    出した中で最良の tar を弾く**（v4 の 195 MB を落としかけた）。
    定数と実測名を直接突き合わせて、書き戻しを止める。
    """
    src = _cells.cell(*INSTALL)
    m = re.search(r"_WC_REQUIRED_KERNELS = \(([^)]*)\)", src)
    assert m, "_WC_REQUIRED_KERNELS の定義が見つからない"
    required = tuple(x.strip().strip("\"'")
                     for x in m.group(1).split(",") if x.strip())
    assert required, "条件が空（＝何でも通る）"
    for name in required:
        assert any(name in real for real in _MEASURED_TAR_KERNELS), (
            f"{name!r} は実測のカーネル名に一致しない: {_MEASURED_TAR_KERNELS}。"
            "実測の tar を弾いてしまう")


def test_warm_cache_require_accepts_a_complete_tar_and_records_the_count(tmp_path):
    """主役が揃っていれば通り、**復元の内訳が `_WC_INFO` に残る**。

    `print` だけだと `cli timings` で横並びにできず、1 回の実行から
    「効いたのか、そもそも当たっていないのか」を読み返せない。
    """
    info = _exec_warm_cache_block(tmp_path, "require",
                                  kernels=_COMPLETE_KERNELS)
    assert info["restored"] is True, info
    assert info["missing"] == [], info
    assert info["entries"] == len(_COMPLETE_KERNELS), info
    assert info["tar"] == "warm-cache.tar", info
    assert info["mode"] == "require", info


def test_warm_cache_restores_an_expanded_dataset(tmp_path):
    """Kaggle は Dataset に上げた tar を**展開して**保持する（実測）。

    2026-11-01 に 195 MB の tar を `datasets create` したら、マウント側は
    `root/.triton/cache/<hash>/...` の木で `*.tar` は 1 本も無かった。
    tar だけを探していると、`require` が**正当な Dataset を落とす**。
    ここが通らないと `KC_EXPERIMENT=cache` は必ず「tar が無い」で死ぬ。
    """
    import os as _os

    info = _exec_warm_cache_block(tmp_path, "require",
                                  kernels=_COMPLETE_KERNELS, tree=True)
    assert info["restored"] is True, info
    assert info["missing"] == [], info
    assert info["entries"] == len(_COMPLETE_KERNELS), info
    assert info["source"].endswith(_os.path.join(".triton", "cache")), info
    # テストのファイルは 1 バイトなので MB は 0 になる。型だけ固定しておく
    # （`timings.json` に書くので、経路によって int/str になると読み手が壊れる）。
    assert isinstance(info["mib"], float), info
    assert info["tar"] == info["source"], info


def test_warm_cache_default_mode_warns_but_continues_on_an_incomplete_tar(tmp_path):
    """既定 `1` は欠けていても落ちない（本番は Dataset 次第で走り続ける）。

    ただし `missing` に残すので、`timings` を見れば「効かなかった」ではなく
    「tar が不完全だった」と後から判る。
    """
    info = _exec_warm_cache_block(tmp_path, "1", kernels=_INCOMPLETE_KERNELS)
    assert info["restored"] is True, info
    assert info["missing"] == ["chunk_fwd_kernel_o",
                               "chunk_gated_delta_rule_fwd_kernel_h"], info


def test_warm_cache_restore_count_reaches_timings():
    """復元の内訳は `timings` まで運ばれている（セルの中で止めていない）。

    `_WC_INFO` を作るだけで `kc_save_timings` に入れ忘れると、ログにしか
    残らず `cli timings` から見えなくなる。
    """
    src = _cells.cell(*INSTALL)
    assert '"warm_cache_restore": dict(_WC_INFO)' in src
    assert src.index("_WC_INFO = {") < src.index('"warm_cache_restore"'), \
        "使う前に定義されていない"


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


# ── 重み prefetch（§22）────────────────────────────────────
def _prefetch_block(src):
    """`LOAD_ARGS` を組み立てるブロックだけを切り出す（PLAN の直前まで）。"""
    start = src.index("# 重みロードの prefetch")
    end = src.index("PLAN = []", start)
    return src[start:end]


def test_prefetch_flags_are_env_gated():
    """重み prefetch のフラグは、env が空なら**1 バイトも足さない**。

    `notes/vllm-startup-speedup.md` §22: vLLM 0.32 の
    `_prefetch_all_checkpoints` は `sorted_files[rank::world_size]` で
    rank ごとにファイルを分担するので、TP=2 でも総読みが 1 回分になる。
    既定のスレッド数 8 は CPU 実測で一番遅い形（8 本 2.3 MiB/s / 1 本 110 MiB/s）。

    既定で足すと対照群（`baseline`）が静かに変わって A/B が壊れるので、
    **env を見て足す**形にしてある。
    """
    from types import SimpleNamespace

    block = _prefetch_block(_cells.cell(*SERVE))

    def load_args(env, supported=True):
        ns = {"os": SimpleNamespace(environ=env),
              "_has": lambda flag: supported,
              "print": lambda *a, **k: None}
        exec(block, ns)          # noqa: S102 - 自前ソースの切り出しのみ
        return ns["LOAD_ARGS"]

    assert load_args({}) == []
    assert load_args({"VLLM_SAFETENSORS_PREFETCH_THREADS": "1"}) == [], (
        "戦略が無いのにスレッドだけ足している")
    assert load_args({"VLLM_SAFETENSORS_STRATEGY": "prefetch",
                      "VLLM_SAFETENSORS_PREFETCH_THREADS": "1"}) == [
        "--safetensors-load-strategy", "prefetch",
        "--safetensors-prefetch-num-threads", "1"]
    # フラグを持たない版なら付けない（黙って落とすと「効かなかった」に見える）
    assert load_args({"VLLM_SAFETENSORS_STRATEGY": "prefetch"}, supported=False) == []


def test_prefetch_is_applied_to_every_attempt():
    """prefetch は読み込み段の話なので、試行段すべてに同じフラグを足す。

    段ごとに変えると、効いたときにどの段のおかげか分からなくなる（v14 の轍）。
    """
    src = _cells.cell(*SERVE)
    assert "_f + LOAD_ARGS" in src, "PLAN の全段に足していない"
    # 足したあとに「7 要素」の検査を通っていること（タプルを壊していない）
    assert src.index("_f + LOAD_ARGS") > src.index(
        'assert all(len(p) == 7 for p in PLAN)')


def test_prefetch_decline_is_detected_and_recorded():
    """vLLM が prefetch を辞退した回を、成果物だけで判別できること。

    実測 2026-11-02（本番 v16）: 同じコード・同じフラグで、`prefetch1` は
    重み 1,081 秒（層あたり 14 秒）、本番は層あたり 47 秒だった。差は
    vLLM の `weight_utils.py:1148` の警告——

        safetensors_load_strategy='prefetch' was explicitly specified, but
        checkpoint total size (18.21 GiB) exceeds 90% of available RAM (6.x GiB)

    ——が出ていたこと。チェックポイント 18.21 GiB に対して RAM 32 GB は
    余裕が無く、**同じ設定でも載るノードと辞退されるノードがある**。
    この警告は 140 KiB のワーカーログの途中に 1 行だけ出るので、拾う側を
    用意しないと「prefetch を付けたのに遅い」を毎回読み違える。
    """
    src = _cells.cell(*SERVE)
    assert '_PF_GUARD = "exceeds 90% of available RAM"' in src, \
        "vLLM の辞退文言を定数にしていない（文言が変わったら気づけるように）"
    assert "if _PF_GUARD in _ln:" in src, "ポーリングで辞退を拾っていない"
    # 記録は result.json（timings）に載せる。ログを落とさないと分からない形は不可。
    assert 'kc_save_timings({"prefetch": _PF_STATE})' in src
    # 「辞退された」ことを起動失敗の文言にも入れる（次に読む人が最初に見る行）
    assert "prefetch={'辞退された' if _PF_GUARD_HIT else '効いていた'}" in src


def test_budget_exhaustion_is_not_reported_as_a_startup_failure():
    """予算切れを「起動に失敗しました」と報告しない。

    実測 2026-11-02（本番 v16）: 予算 4,200 秒で切られたのに result.json の
    error は「vLLM の起動に失敗しました」だった。設定の誤りを疑って
    ログを 1 時間掘ることになった（実際は 62 層中 56 層まで読めていた）。
    待つのをやめた理由を `_why` に残し、文言を分ける。
    """
    src = _cells.cell(*SERVE)
    for branch in ('_why = "started"', '_why = "died"', '_why = "budget"'):
        assert branch in src, f"待機の終了理由 {branch} を残していない"
    # 予算切れの枝が先にあり、そこでは「失敗」と言わない
    i_budget = src.index('if _why == "budget":')
    i_fail = src.index('fail("vLLM の起動に失敗しました', i_budget)
    assert i_budget < i_fail
    budget_msg = src[i_budget:i_fail]
    assert "予算" in budget_msg and "設定の誤りではない" in budget_msg
    assert "起動に失敗" not in budget_msg


# ── prefetch 可否の判定（v17 の 6,128 秒）─────────────────────
def _ram_block(src):
    """RAM を見る関数だけを切り出す（`_mem_line` の直前まで）。"""
    return src[src.index("def _mem_info("):src.index("def _mem_line(")]


def test_prefetch_decision_uses_the_same_formula_as_vllm():
    """prefetch の可否は vLLM と同じ式で判定する。

    実測 2026-11-02（本番 v17）: こちらの `MemAvailable` は 30,159 MiB で
    「足りている」と出したのに、vLLM は `Available RAM: 6.16 GiB` と判定して
    prefetch を辞退した。理由は `vllm/utils/cpu_resource_utils.py` の
    `get_available_memory` が

        min(psutil.virtual_memory().available, cgroup 上限 - 使用量)

    で、**cgroup の使用量に page cache が含まれる**ため。このノードは
    page cache が 25 GiB 張り付いており、32,098 − 約 25,800 ≒ 6.3 GiB。
    辞退されると GCSFUSE 上の safetensors をテンソル単位で読む経路に落ち、
    18.21 GiB に **6,128 秒**（3.2 MB/s）かかった。同じノードの逐次読みは
    110〜130 MiB/s なので、判定を間違えると 5 倍の時間を焼く。
    """
    src = _cells.cell(*SERVE)
    assert "def _cgroup_mib(" in src, "cgroup を読んでいない"
    assert "/sys/fs/cgroup/memory.max" in src, "上限を読んでいない"
    assert "/sys/fs/cgroup/memory.current" in src, "使用量を読んでいない"
    assert "def _prefetch_avail_mib(" in src, "vLLM と同じ式の関数が無い"
    block = src[src.index("def _prefetch_avail_mib("):src.index("def _mem_line(")]
    assert "min(avail, limit - used)" in block, "min() で束ねていない"
    # 試行ループのゲートも同じ式を通す（片方だけ直す事故を防ぐ）
    loop = src[src.index("for idx, (label, spec, maxlen"):]
    assert "_avail = _prefetch_avail_mib()" in loop, \
        "ゲートが MemAvailable のまま（vLLM と食い違う）"


def test_prefetch_avail_is_the_minimum_of_memavailable_and_cgroup():
    """式そのものを動かして確かめる（`min` の向きを間違えても気づける形で）。"""
    ns = {}
    exec(_ram_block(_cells.cell(*SERVE)), ns)      # noqa: S102 - 自前ソースのみ

    # 実測 v17 の数字（MemAvailable 30,159 / 上限 32,098 / 使用 25,800）
    ns["_mem_info"] = lambda: {"MemAvailable": 30159}
    ns["_cgroup_mib"] = lambda: (32098, 25800)
    assert ns["_prefetch_avail_mib"]() == 32098 - 25800, \
        "cgroup の空きで頭打ちになっていない"
    # cgroup に余裕があれば MemAvailable が答え
    ns["_cgroup_mib"] = lambda: (32098, 1000)
    assert ns["_prefetch_avail_mib"]() == 30159
    # cgroup が読めない環境では従来どおり MemAvailable
    ns["_cgroup_mib"] = lambda: (None, None)
    assert ns["_prefetch_avail_mib"]() == 30159
    # 使用量が上限を超えていても負にしない
    ns["_cgroup_mib"] = lambda: (32098, 99999)
    assert ns["_prefetch_avail_mib"]() == 0


def test_prewarm_turns_on_by_itself_only_when_prefetch_will_be_declined():
    """辞退される見込みの回だけ、自前の逐次読みに切り替える。

    v17 は `KC_PREWARM` 無効のまま 6,128 秒を焼いた。同じノードでも 1 本の
    逐次読みは桁が違うので、**辞退が確実な回に限り**自動で試す。ただし
    先読みは `_prewarm_weights` が probe で速度を測り、予算に収まらなければ
    **始まらない**（遅いノードで時間だけ払う形にしない）。
    `KC_PREWARM=0` の明示は自動も止める（A/B の対照群を守る）。
    """
    src = _cells.cell(*SERVE)
    assert '_pw_auto = (_pw_mode in ("", "auto")) and not _PF_OK' in src, \
        "自動で立つ条件が無い"
    assert '_pw_mode not in ("", "auto", "0") or _pw_auto' in src, \
        "KC_PREWARM=0 で止められない"
    assert '"decided_by"' in src, "どちらで立ったかが残らない"
    # 予算に収まらない見込みなら始めない（probe → over budget）
    assert "over budget" in src


def test_declined_prefetch_reports_an_expected_load_time():
    """辞退されたときに「このままだと何秒か」を出す。

    v17 は辞退されたあと 6,128 秒を無言で焼き、予算 7,000 秒を超えて
    「起動失敗」に見える形で落ちた。先読みの実測レートがあれば、
    重みの残り時間は割り算で出せる。
    """
    src = _cells.cell(*SERVE)
    assert "prefetch 用の空き" in src, "辞退の理由（式の内訳）を出していない"
    assert "予算 {STARTUP_BUDGET}s に収まるか確認" in src, \
        "予算に入るかの警告が無い"
