"""spec（kbuild/spec.py）からビルダーを引いて生成する。

    python3 -m cli build              # active 全部
    python3 -m cli build exl3-serve   # 指定のみ

ビルダーは `kaggle/build_*.py` をサブプロセスで実行する（従来どおり）。
生成物は各ビルダーが自分のディレクトリへ書く。
"""
import os
import pathlib
import subprocess
import sys

from kbuild import spec

ROOT = spec.ROOT

# `--smoke` を受けるビルダーと、`--smoke` で push してよいカーネル。
#
# **手書きしない。** 台帳（`kbuild/spec.py` の `smoke`）から導出する。
# 以前はここに 6 本ずつ直書きしていて、新カーネルを足すと**静かに漏れた**
# （`smoke` を渡しても「対象外」で止まるので気づきにくい）。
# smoke 版の出力先は `smoke_out`（空なら同じディレクトリ）。push は
# そちらを staging するので通常版を上書きしない。
SMOKE_KERNELS = frozenset(
    k.name for k in spec.KERNELS if k.smoke and k.status != spec.RETIRED)
SMOKE_BUILDERS = frozenset(
    k.builder for k in spec.KERNELS
    if k.smoke and k.status != spec.RETIRED and k.builder)


def plan(targets=None):
    """(ビルダー, env) の実行計画。"""
    return spec.builders(targets)


def _extra_args():
    """`KC_BUILD_ARGS`（空白区切り）をビルダーの追加引数にする。"""
    return os.environ.get("KC_BUILD_ARGS", "").split()


def run(targets=None, dry_run=False, extra_args=None):
    plan_ = plan(targets)
    if not plan_:
        print("ビルダーを持つカーネルが無い（ビルダー無しの台帳のみ）")
        return 0
    extra = _extra_args() if extra_args is None else list(extra_args)
    failed = []
    for builder, env in plan_:
        path = pathlib.Path(ROOT) / "kaggle" / f"{builder}.py"
        if not path.is_file():
            failed.append(builder)
            print(f"NG  {builder}: {path} が無い")
            continue
        args = extra if builder in SMOKE_BUILDERS else []
        label = builder + (f" [{', '.join(f'{k}={v}' for k, v in env.items())}]"
                           if env else "")
        if args:
            label += f" {' '.join(args)}"
        if dry_run:
            print(f"DRY {label}")
            continue
        run_env = dict(os.environ)
        run_env.update(env)
        p = subprocess.run([sys.executable, str(path), *args],
                           cwd=ROOT, env=run_env)
        print(f"{'OK ' if p.returncode == 0 else 'NG '} {label}")
        if p.returncode != 0:
            failed.append(builder)
    return 1 if failed else 0
