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
    env_at = src.index("os.environ.setdefault('KC_WARM_CACHE', '1')")
    cell_at = src.index("Step 2 (vLLM): 配信カーネルの設定")
    assert env_at < cell_at, "設定セルより後に環境変数を入れている"


def test_every_experiment_declares_both_knobs():
    """全実験が `KC_PREWARM` と `KC_WARM_CACHE` を**明示**する。

    片方だけ既定に頼ると、既定を変えた瞬間に対照群が壊れる。v14 で
    「先読みのせい」と「既定が変わったせい」が区別できなくなったのと同じ形。
    """
    mod = _module()
    assert set(mod.EXPERIMENTS) == {
        "baseline", "prewarm_on", "cache", "force_first"}, mod.EXPERIMENTS
    for name, (desc, env) in mod.EXPERIMENTS.items():
        assert "KC_PREWARM" in env, f"{name} が KC_PREWARM を明示していない"
        assert "KC_WARM_CACHE" in env, f"{name} が KC_WARM_CACHE を明示していない"
        assert desc, f"{name} に説明が無い"


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
