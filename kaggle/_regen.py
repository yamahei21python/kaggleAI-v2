"""ビルダーを一時ディレクトリで再生成する共通処理。

`check_notebooks.py` / `check_scripts.py` は「commit 済みの生成物がビルダーの
出力と一致するか」を見る。ビルダーはリポジトリ内へ直接書き込むため、
作業ツリーを汚さないよう一時ディレクトリへ写してから走らせる。

ここが持つのは**写す・走らせる・比べる・結果を出す**の 4 つだけ。
どのカーネルを見るかは呼び出し側（各チェッカー）が決める。
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ビルダーが必要とするもの（kbuild/ runtime/ cli/ を import する）
COPY_FILES = ("llama-chat.html", "config.env")
COPY_DIRS = ("kaggle", "kbuild", "runtime", "cli")
IGNORE = shutil.ignore_patterns("archive", "__pycache__")

# 環境変数はビルダーのノブ。呼び出し側が渡す分だけを有効にする。
ENV_PREFIXES = ("STEP2", "KC_")


def clean_env(extra=None) -> dict:
    """ビルダーへ渡す環境変数（ノブの取りこぼしを防ぐため接頭辞で落とす）。"""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(ENV_PREFIXES)}
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _clear_outputs(work):
    """写した生成物を消す。

    消さないと「ビルダーが書けなかった」ときに、写しただけの古いファイルが
    一致して**偽の成功**になる。生成物のファイル名だけを消す（ディレクトリは
    セル置き場などを巻き込むので触らない）。

    台帳（kbuild/spec.py）が無い古いブランチでもこのモジュールを import
    できるよう、台帳の読み込みだけ任意にしてある。その場合ビルダー自身が
    kbuild を要求して失敗するので、偽の成功にはならない。
    """
    try:
        from kbuild import spec
    except ImportError:
        return
    for k in spec.active():
        if not k.builder:
            continue
        dirs = [k.dir]
        if k.smoke_out:
            dirs.append(k.smoke_dir)      # smoke 版は別ディレクトリに出る
        for d in dirs:
            for name in (k.code_file, k.metadata_file):
                p = pathlib.Path(work) / d.relative_to(ROOT) / name
                if p.is_file():
                    p.unlink()


def stage(work) -> pathlib.Path:
    """一時ディレクトリへリポジトリの必要部分を写し、生成物を消す。

    `config.local.env` は写さない。コミットする生成物は `config.env`（既定値）
    で作ったものが基準で、個人値は Kaggle 側の Secrets で上書きする前提。
    """
    work = pathlib.Path(work)
    for name in COPY_FILES:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, work / name)
    for name in COPY_DIRS:
        shutil.copytree(ROOT / name, work / name, ignore=IGNORE)
    _clear_outputs(work)
    return work


def run(work, builder: str, extra_env=None):
    """ビルダーを work 内で実行する（builder は "build_step2" 形式）。"""
    script = builder if builder.endswith(".py") else f"{builder}.py"
    return subprocess.run(
        [sys.executable, str(pathlib.Path(work) / "kaggle" / script)],
        cwd=str(work), env=clean_env(extra_env),
        capture_output=True, text=True)


def label(builder: str, extra_env=None) -> str:
    """ログ用の見出し（ビルダー + 渡したノブ）。"""
    extra = extra_env or {}
    return builder + "".join(f" {k}={v}" for k, v in extra.items())


def diff(work, rel: str):
    """(状態, 詳細) を返す。状態は "same" / "missing" / "uncommitted" / "diff"。

    `rel` はリポジトリ相対（例 "kaggle/step1/step1-....ipynb"）。
    """
    fresh = pathlib.Path(work) / rel
    committed = ROOT / rel
    if not fresh.is_file():
        return "missing", "生成されていない"
    if not committed.is_file():
        return "uncommitted", "未生成（生成してコミットする）"
    if fresh.read_bytes() == committed.read_bytes():
        return "same", ""
    return "diff", "再生成してください"


def report(rel: str, state: str, detail: str, tag: str) -> bool:
    """1 生成物ぶんの結果を出し、不一致なら True（＝失敗）を返す。

    notebook 版も script 版も同じ体裁で出したいので、文言はここに置く。
    """
    if state == "same":
        print(f"✅ {rel}")
        return False
    if state == "diff":
        print(f"❌ {rel}  ← {tag} で再生成してください")
    else:
        print(f"❌ {rel}: {detail}")
    return True
