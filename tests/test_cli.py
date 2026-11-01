"""CLI の既定動作を検査する（ネットワーク・Kaggle 不要）。

要点は「既定は build のみ」「push は --yes が無いと走らない」。
"""
import json
import pathlib

from cli import __main__ as cli_main
from cli import build as build_mod
from cli import push as push_mod
from kbuild import spec


def test_default_is_build(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(build_mod, "run",
                        lambda targets=None, dry_run=False, extra_args=None:
                        calls.update(targets=targets, dry_run=dry_run) or 0)
    assert cli_main.main([]) == 0
    assert calls == {"targets": None, "dry_run": False}


def test_push_requires_yes(monkeypatch, capsys):
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: ["GPU: 残り 1h / 上限 30h"])
    rc = push_mod.push("exl3-serve", yes=False)
    out = capsys.readouterr().out
    assert rc == 1
    assert "--yes" in out


def test_pull_plan_dedup():
    plan = build_mod.plan()
    assert plan
    assert all(isinstance(env, dict) for _, env in plan)


def test_default_plan_excludes_retired():
    builders = {b for b, _ in build_mod.plan()}
    assert "build_capture_patch" not in builders


def test_push_refuses_retired(monkeypatch):
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    assert push_mod.push("keepalive-test-1", yes=True) == 1


def test_smoke_only_passed_to_capable_builders(capsys):
    build_mod.run(dry_run=True, extra_args=["--smoke"])
    out = capsys.readouterr().out
    assert "build_engine_serve --smoke" in out
    assert "build_exl3_serve --smoke" in out
    assert "build_ci_cpu --smoke" not in out


def test_push_refuses_smoke_for_unsupported_kernel(monkeypatch):
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    for name in ("exl3-serve-dryrun", "keepalive-test"):
        assert push_mod.push(name, yes=True, smoke=True) == 1


def test_push_accepts_smoke_for_serve_kernels(monkeypatch):
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    for name in sorted(build_mod.SMOKE_KERNELS):
        assert push_mod.push(name, yes=False, smoke=True) == 1   # --yes 待ちで止まる
        assert spec.find(name).status != spec.RETIRED


def test_push_smoke_stages_smoke_output(monkeypatch, capsys):
    """push --smoke は通常版ではなく smoke 専用出力を staging する。"""
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    kernel = spec.find("step2-vllm-qwen38-27b-cloudflared")
    assert kernel.smoke_dir != kernel.dir
    assert push_mod.push(kernel.name, yes=True, dry_run=True, smoke=True) == 0
    out = capsys.readouterr().out
    assert str(kernel.smoke_dir.relative_to(spec.ROOT)) in out


def test_ambiguous_name_is_rejected():
    import pytest
    with pytest.raises(KeyError, match="候補が複数"):
        spec.find("step2")


def test_push_no_run_is_save_only(monkeypatch, capsys):
    """--no-run は実行しないので、クォータを読まず「保存のみ」と明示する。"""
    def boom():
        raise AssertionError("--no-run なのにクォータを読んだ")
    monkeypatch.setattr(push_mod, "_quota_lines", boom)
    rc = push_mod.push("llama-server-ngrok-script", yes=False, no_run=True)
    out = capsys.readouterr().out
    assert rc == 1
    assert "保存のみ" in out and "Quick Save" in out


def test_push_no_run_dry_run_shows_flag(monkeypatch, capsys):
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    rc = push_mod.push("llama-server-ngrok-script", yes=True, dry_run=True, no_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "--no-run" in out and "[no-run]" in out


def test_push_no_run_needs_cli_2x(monkeypatch, capsys):
    """2.x が無いときは「作る方法」まで出す（黙って実行しない）。"""
    monkeypatch.setattr(push_mod, "_quota_lines", lambda: [])
    monkeypatch.setattr(push_mod, "kaggle_exe", lambda need_no_run=False: None)
    rc = push_mod.push("llama-server-ngrok-script", yes=True, no_run=True)
    out = capsys.readouterr().out
    assert rc == 1
    assert "kaggle CLI 2.x" in out and ".venv-kaggle-cli" in out


def test_stage_renames_metadata(tmp_path):
    kernel = spec.find("keepalive-test")
    assert kernel.metadata_file != "kernel-metadata.json"   # 例外ケース
    push_mod._stage(kernel, tmp_path)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"] == kernel.id
    assert (tmp_path / kernel.code_file).is_file()


def test_stage_requires_generated_code(tmp_path):
    kernel = spec.find("exl3-serve")
    fake = pathlib.Path(tmp_path) / "serve"
    fake.mkdir()
    missing = type(kernel)(**{**kernel.__dict__, "out": str(fake)})
    try:
        push_mod._stage(missing, tmp_path / "stage")
    except SystemExit as e:
        assert "生成物が無い" in str(e)
    else:                                                      # pragma: no cover
        raise AssertionError("生成物が無いのに通った")
