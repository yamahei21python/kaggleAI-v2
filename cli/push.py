"""カーネルを push する（明示操作）。

push は Kaggle 側で即実行され、GPU カーネルはクォータを消費する。
そのため既定では走らせず、`--yes` を要求する。実行前に残量を表示する。

    python3 -m cli push exl3-serve --yes
    python3 -m cli push llama-server-ngrok-script --no-run --yes   # 作成だけ

`--no-run` は **Quick Save 相当**（バージョンを保存するだけで 1 セルも実行しない）。
GPU クォータを消費せずにカーネル id だけ先に確保したいときに使う。ただし
`kaggle` CLI 2.x にしか無いフラグなので、1.x しか無ければ理由を出して止まる。

`kaggle kernels push` は `kernel-metadata.json` という固定名を読むため、
メタデータ名が違うカーネル（keepalive-test）は一時ディレクトリで組み立てる。
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from kbuild import spec

# 2.x の CLI をリポジトリ内に置いておくための場所（`.gitignore` 済み）。
# 1.x は `--no-run` を持たないので、既定の `kaggle` とは別に持つ。
KAGGLE_2X = spec.ROOT / ".venv-kaggle-cli" / "bin" / "kaggle"


def _supports(exe, flag):
    """その CLI が `kernels push` で flag を受けるか（--help を読むだけ）。"""
    try:
        p = subprocess.run([exe, "kernels", "push", "--help"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (p.stdout or "")


def kaggle_exe(need_no_run=False):
    """使う `kaggle` を決める。

    `--no-run` が要るときだけ 2.x を探す（1.x は持っていない）。順番は
    `KC_KAGGLE` → リポジトリ内 venv → PATH の `kaggle` → `kaggle2`。
    """
    cands = [os.environ.get("KC_KAGGLE"), str(KAGGLE_2X),
             shutil.which("kaggle"), shutil.which("kaggle2")]
    for exe in cands:
        if not exe or not pathlib.Path(exe).exists():
            continue
        if not need_no_run or _supports(exe, "--no-run"):
            return exe
    return None


def _quota_lines():
    """GPU/TPU 残量（取れなければ理由を 1 行で返す）。"""
    try:
        from cli import quota as q
    except Exception as e:                                  # pragma: no cover
        return [f"クォータ取得不可: {e}"]
    try:
        data = q.fetch()
    except SystemExit as e:
        return [f"クォータ取得不可: {e.code}"]
    except Exception as e:
        return [f"クォータ取得不可: {type(e).__name__}: {e}"]
    lines = []
    for label, key in (("GPU", "gpuQuota"), ("TPU", "tpuQuota")):
        quota = data.get(key) or {}
        used = quota.get("timeUsed", {}).get("seconds", 0)
        total = quota.get("totalTimeAllowed", {}).get("seconds", 0)
        reserved = quota.get("timeReserved", {}).get("seconds", 0)
        left = total - used - reserved
        lines.append(f"{label}: 残り {q.hms(left)} / 上限 {q.hms(total)}")
    return lines


def _rel(path):
    """ログ用にリポジトリ相対へ（外なら絶対パスのまま）。"""
    try:
        return pathlib.Path(path).relative_to(spec.ROOT)
    except ValueError:
        return pathlib.Path(path)


def _stage(kernel, tmp, src_dir=None):
    """code_file とメタデータを push 用の一時ディレクトリに揃える。

    `src_dir` を渡すと通常版ではなくその場所（smoke 専用出力）から拾う。
    """
    tmp = pathlib.Path(tmp)
    src_dir = pathlib.Path(src_dir) if src_dir else kernel.dir
    code = src_dir / kernel.code_file
    if not code.is_file():
        raise SystemExit(f"生成物が無い: {code}\n先に `python3 -m cli build` を実行")
    shutil.copy2(code, tmp / kernel.code_file)
    meta = src_dir / kernel.metadata_file
    if not meta.is_file():
        raise SystemExit(f"メタデータが無い: {meta}")
    data = json.loads(meta.read_text(encoding="utf-8"))
    (tmp / "kernel-metadata.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def push(name, yes=False, dry_run=False, smoke=False, no_run=False):
    kernel = spec.find(name)
    if kernel.status == spec.RETIRED:
        print(f"{kernel.id} は退役。push しない（台帳の status を確認）")
        return 1
    from cli import build as build_mod
    if smoke and kernel.name not in build_mod.SMOKE_KERNELS:
        print(f"{kernel.id} は smoke 非対応（対象: "
              + ", ".join(sorted(build_mod.SMOKE_KERNELS)) + ")")
        return 1
    # smoke は通常版と別ディレクトリに生成されるので、staging 元も分ける。
    src_dir = kernel.smoke_dir if smoke else kernel.dir
    print(f"対象: {kernel.id} ({kernel.kind}, gpu={kernel.gpu})"
          + ("  [smoke]" if smoke else "")
          + ("  [no-run]" if no_run else ""))
    print(f"元  : {_rel(src_dir)}")
    if no_run:
        print("モード: 保存のみ（--no-run = Quick Save。1 セルも実行せず、GPU を消費しない）")
    else:
        for line in _quota_lines():
            print(line)
    if not yes:
        if no_run:
            print("\nバージョンを保存するだけ（実行しない）。進めるなら --yes を付ける。")
        else:
            print("\npush は即実行されクォータを消費する。実行するなら --yes を付ける。")
        return 1
    if dry_run:
        print(f"DRY kaggle kernels push -p {src_dir}" + (" --no-run" if no_run else ""))
        return 0
    # CLI の解決はここまで遅らせる（--no-run の可否を毎回 subprocess で見るため）。
    exe = kaggle_exe(need_no_run=no_run)
    if not exe:
        if no_run:
            print("--no-run を使うには kaggle CLI 2.x が要る（1.7.4.5 には無い）。")
            print(f"  用意する場所: {_rel(KAGGLE_2X)}")
            print("  作るには: python3.12 -m venv .venv-kaggle-cli && "
                  ".venv-kaggle-cli/bin/pip install kaggle")
            print("  もしくは KC_KAGGLE=/path/to/kaggle2 を指定する。")
        else:
            print("kaggle CLI が見つからない（PATH を確認）")
        return 1
    # 古い生成物を本番投入しないよう、push 直前に再生成する。
    rc = build_mod.run([kernel.name], extra_args=["--smoke"] if smoke else [])
    if rc != 0:
        print("生成に失敗したため push を中止")
        return rc
    with tempfile.TemporaryDirectory(prefix="kc-push-") as tmp:
        _stage(kernel, tmp, src_dir)
        cmd = [exe, "kernels", "push", "-p", str(tmp)]
        if no_run:
            cmd.append("--no-run")
        p = subprocess.run(cmd)
    return p.returncode


if __name__ == "__main__":                                  # pragma: no cover
    argv = sys.argv[1:]
    sys.exit(push(argv[0], yes="--yes" in argv, no_run="--no-run" in argv))
