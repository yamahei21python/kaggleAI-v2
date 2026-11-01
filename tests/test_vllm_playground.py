"""実験用 playground カーネル（`vllm-playground`）の骨組みを固定する。

**本番と同じセル**（`kaggle/step2-vllm/cells/*.py`）を、公開系（トンネル /
proxy / keepalive）だけ外した骨組みで動かすのが要点。ここだけ別実装にすると
「playground では効いたのに本番では効かない」が起きる。

`status=SEMI` なので `cli build` / `check_scripts` の対象外。代わりにここで
骨組みを見る（`KC_EXPERIMENT` ごとに生成物が変わるので、バイト一致では
固定できない）。
"""
import importlib.util
import json
import os
import pathlib
import py_compile
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILDER = ROOT / "kaggle" / "build_vllm_playground.py"


def _build(tmp_path, experiment="baseline"):
    """ビルダーを一時ディレクトリへ走らせて (出力先, 完了結果) を返す。"""
    out = tmp_path / "out"
    env = {**os.environ,
           "KC_PLAYGROUND_OUT": str(out),
           "KC_EXPERIMENT": experiment}
    p = subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return out, p


def _module():
    """ビルダーをモジュールとして読む（`kaggle` は PyPI と衝突するため）。"""
    spec = importlib.util.spec_from_file_location("build_vllm_playground",
                                                  BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_playground_has_its_own_kernel_id(tmp_path):
    """実験は本番のバージョン履歴を汚さない（別 id）。

    それまでは `--smoke` で本番カーネルへ push していた。smoke 変種は
    **同じ kernel id** なので、実験のたびに本番の版が埋まる。v15 もそうやって
    本番の版として積まれている。
    """
    out, p = _build(tmp_path)
    assert p.returncode == 0, (p.stdout + p.stderr)[-900:]
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "yamahei21python/vllm-playground"
    assert meta["enable_gpu"] is True
    assert meta["kernel_type"] == "script"
    assert "step2-vllm" not in meta["id"], "本番カーネルを指している"


def test_playground_attaches_the_warm_cache_dataset(tmp_path):
    """キャッシュ Dataset は付けたまま、復元は `KC_WARM_CACHE` で切る。

    付け外しで A/B すると、Dataset の有無という別の変数が混ざる。
    """
    out, _ = _build(tmp_path)
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert "yamahei21python/vllm-warm-cache" in meta["dataset_sources"]


def test_experiment_env_is_set_before_the_settings_cell(tmp_path):
    """実験の環境変数は設定セルより**前**に置く。

    後ろだと、設定セルや install セルが既定値で動き出してから書き換えることに
    なり、ノブが効かない。そうなると「playground では効いた」が嘘になる。
    """
    out, p = _build(tmp_path, "cache")
    assert p.returncode == 0, (p.stdout + p.stderr)[-900:]
    src = (out / "main.py").read_text(encoding="utf-8")
    # 値は実験表から引く（直書きすると、ノブの値を変えるたびにテストが腐る）。
    value = _module().EXPERIMENTS["cache"][1]["KC_WARM_CACHE"]
    env_at = src.index(f"os.environ.setdefault('KC_WARM_CACHE', {value!r})")
    cell_at = src.index("Step 2 (vLLM): 配信カーネルの設定")
    assert env_at < cell_at, "設定セルより後に環境変数を入れている"


def test_cache_experiment_requires_the_dataset():
    """`cache` は `KC_WARM_CACHE=require`（無ければ落ちる）で走らせる。

    `1` のままだと、Dataset が未作成／空でも続行し、**baseline とまったく同じ
    数字**が出る。すると「キャッシュは効かない」という偽陰性になり、1 往復
    （約 40 分）を使って間違った結論に到達する。実測 2026-11-01 時点で
    `vllm-warm-cache` は未作成なので、この設定が無いと実際に踏んでいた。
    """
    mod = _module()
    assert mod.EXPERIMENTS["cache"][1]["KC_WARM_CACHE"] == "require", (
        "cache 実験が require でない。Dataset 未作成でも走って偽陰性になる。")
    # 対照群（baseline）は 0 のままであること＝A/B が成立する
    assert mod.EXPERIMENTS["baseline"][1]["KC_WARM_CACHE"] == "0"


def test_every_experiment_declares_both_knobs():
    """全実験が `KC_PREWARM` と `KC_WARM_CACHE` を**明示**する。

    片方だけ既定に頼ると、既定を変えた瞬間に対照群が壊れる。v14 で
    「先読みのせい」と「既定が変わったせい」が区別できなくなったのと同じ形。
    """
    mod = _module()
    assert set(mod.EXPERIMENTS) == {
        "baseline", "prewarm_on", "cache", "force_first",
        "baseline_long", "prefetch1"}, mod.EXPERIMENTS
    for name, (desc, env) in mod.EXPERIMENTS.items():
        assert "KC_PREWARM" in env, f"{name} が KC_PREWARM を明示していない"
        assert "KC_WARM_CACHE" in env, f"{name} が KC_WARM_CACHE を明示していない"
        assert desc, f"{name} に説明が無い"


def test_baseline_long_only_moves_the_budget():
    """`baseline_long` は予算だけ動かす（構成は `baseline` と同一）。

    2026-11-01 の `cache` は復元に成功したのに、試行 1 が重みロードだけで
    4,200 秒を使い切り、Triton キャッシュが効くはずのコンパイル段に届かな
    かった。予算を上げずに `cache` を回し直しても同じ場所で切れるので、
    「重みロードが遅い」を先に単独で解くための実験。

    ここが 2 ノブになると、通ったときに「予算のおかげ」と「構成のおかげ」が
    区別できない（v14 と同じ轍）。
    """
    mod = _module()
    base = mod.EXPERIMENTS["baseline"][1]
    long_ = mod.EXPERIMENTS["baseline_long"][1]

    budget_keys = {"VLLM_STARTUP_BUDGET", "VLLM_ATTEMPT_TIMEOUT"}
    assert set(long_) - set(base) == budget_keys, (
        f"予算以外のノブが増えている: {set(long_) - set(base) - budget_keys}")
    assert {k: v for k, v in long_.items() if k not in budget_keys} == base

    # 試行の上限は `min(ATTEMPT_TIMEOUT, 残予算)`。総予算だけ上げて
    # ATTEMPT_TIMEOUT を据え置くと、試行 1 は同じ秒数で切られる。
    assert int(long_["VLLM_STARTUP_BUDGET"]) == 7000
    assert int(long_["VLLM_ATTEMPT_TIMEOUT"]) == int(long_["VLLM_STARTUP_BUDGET"])

    # 4,200 の側は動かしていないこと（比較表が壊れない）
    assert "VLLM_STARTUP_BUDGET" not in base


def test_baseline_is_the_out2g_configuration():
    """`baseline` は out2g（総 928.7s / 31.7 t/s）の構成を指す。

    先読み OFF・キャッシュ復元なし。ここが動くと「基準」が動く。
    """
    env = _module().EXPERIMENTS["baseline"][1]
    assert env["KC_PREWARM"] == "0"
    assert env["KC_WARM_CACHE"] == "0"


def test_unknown_experiment_is_rejected(tmp_path):
    """未知のノブ名は**ビルド時に**落とす。

    カーネルを回してから「そんな実験は無い」と分かるのは 30 分の無駄。
    """
    _out, p = _build(tmp_path, "nope")
    assert p.returncode != 0
    assert "KC_EXPERIMENT" in (p.stdout + p.stderr)


def test_playground_runs_without_a_tunnel(tmp_path):
    """トンネルを張らない骨組みで生成される（`smoke=True` と組のときだけ許す）。"""
    out, p = _build(tmp_path, "force_first")
    assert p.returncode == 0, (p.stdout + p.stderr)[-900:]
    src = (out / "main.py").read_text(encoding="utf-8")
    assert 'TUNNEL = "none"' in src
    assert "SMOKE = True" in src
    # force_first のノブが実際に入っていること（表に書いただけになっていない）
    assert "VLLM_TRITON_FORCE_FIRST_CONFIG" in src
    py_compile.compile(str(out / "main.py"), doraise=True)


def test_smoke_returns_before_the_keepalive_loop(tmp_path):
    """smoke は `loop()` に入る**前**に return する。

    `RUN_MINUTES=60` は keepalive ループの予算で、`loop()` は
    `started = time.time()` を**入った時点**で取る。もし smoke の return が
    `loop()` より後ろへ動くと、起動に 7,000 秒かける実験（`baseline_long`）が
    サーブ段で 60 分待たされ、合計が Kaggle の上限に近づく。
    2026-11-02 01:20 に「7,000 秒が別の上限に殺されないか」を確認した記録。
    """
    out, p = _build(tmp_path, "baseline_long")
    assert p.returncode == 0, (p.stdout + p.stderr)[-900:]
    src = (out / "main.py").read_text(encoding="utf-8")
    # `reason = loop(procs)` は keepalive シナリオ側にもあるので、**本番経路**
    # （最後の出現）と比べる。最初の出現と比べると常に「後ろ」に見える。
    assert src.index('return "smoke"') < src.rindex("reason = loop(procs)"), (
        "smoke の return が本番経路の keepalive ループより後ろにある")
    # 予算が実際に効いていること（表に書いただけになっていない）
    assert "os.environ.setdefault('VLLM_STARTUP_BUDGET', '7000')" in src


def test_tunnel_none_is_rejected_outside_smoke():
    """本番（smoke=False）で `tunnel="none"` を渡すのは設定ミス。弾く。

    playground のために緩めた条件が、本番の取り違えまで通してしまわないこと。
    """
    import pytest

    from kbuild import serve_script
    with pytest.raises(ValueError):
        serve_script.render(
            {"name": "x", "alias": "y"}, tunnel="none",
            run_minutes=1, runtime="", proxy_b64="", html_b64="",
            gist_id="", gist_owner="")


def test_prefetch1_is_one_knob_against_baseline_long():
    """`prefetch1` は `baseline_long` と予算まで揃え、**prefetch だけ**足す。

    予算を揃えないと「prefetch の効果」と「予算の効果」が混ざる。
    2026-11-02 01:58 の `baseline_long`（3,820 秒 / 重み 2,802 秒）が対照。
    """
    mod = _module()
    base = mod.EXPERIMENTS["baseline_long"][1]
    pf = mod.EXPERIMENTS["prefetch1"][1]

    added = set(pf) - set(base)
    assert added == {"VLLM_SAFETENSORS_STRATEGY",
                     "VLLM_SAFETENSORS_PREFETCH_THREADS"}, added
    assert {k: v for k, v in pf.items() if k not in added} == base
    assert pf["VLLM_SAFETENSORS_STRATEGY"] == "prefetch"
    # 既定の 8 は実測で一番遅い形（8 本 2.3 MiB/s / 1 本 110 MiB/s）
    assert pf["VLLM_SAFETENSORS_PREFETCH_THREADS"] == "1"
