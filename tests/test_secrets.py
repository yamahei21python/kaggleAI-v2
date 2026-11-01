"""Secrets の参照と台帳（kbuild/secrets.py）が一致するか確認する。

台帳に無い Secret を参照したままデプロイすると、カーネルは起動後に
初めて「Secret 未登録」で落ちる。静的に検出できるものはここで止める。
"""
import pathlib
import re

from kbuild import secrets

ROOT = pathlib.Path(__file__).resolve().parent.parent
GET_RE = re.compile(r'get_secret\(\s*"([A-Z0-9_]+)"')
SECRET_RE = re.compile(r'(?<![A-Za-z_])secret\(\s*"([A-Z0-9_]+)"')
# エンジン辞書（serve_script の engine_code / build_exl3_serve の _engine）は
# Secret 名をデータで持つ。リテラル参照と同じように台帳と突き合わせる。
SECRET_KEY_RE = re.compile(r'"secret_(?:user|pass)"\s*:\s*"([A-Z0-9_]+)"')
GENERATED = {
    ROOT / "kaggle" / "exl3-serve-kernel" / d
    for d in ("serve", "dryrun", "keepalive")
} | {
    ROOT / "kaggle" / "lmdeploy-serve",
    ROOT / "kaggle" / "sglang-serve",
    # Ollama / vLLM の script 版（serve_script の生成物。Secret 参照は
    # kbuild/serve_script.py 側で台帳に載っている）
    ROOT / "kaggle" / "step2-script",
    ROOT / "kaggle" / "step2-vllm-script",
    ROOT / "kaggle" / "llama-server-script",
    # 実験用 playground（同じく serve_script の生成物）。トンネル / Gist /
    # keepalive を丸ごと飛ばすので、GIST_ID / NGROK_TOKEN 等の参照は
    # テンプレートに残っているが**到達しない**。二重に数えない。
    ROOT / "kaggle" / "vllm-playground",
    # gist-probe は runtime/gist.py をそのまま埋め込んだ生成物。参照は
    # 真実源（runtime/gist.py）側で台帳に載っている。
    ROOT / "kaggle" / "cpu-tests" / "gist",
}


def _names(path):
    src = path.read_text(encoding="utf-8")
    return {m.group(1) for pat in (GET_RE, SECRET_RE, SECRET_KEY_RE)
            for m in pat.finditer(src)}


def _is_generated(path):
    if any(str(path).startswith(str(d)) for d in GENERATED):
        return True
    # 配信しないカーネルの script 版（kaggle/<kernel>/script/main.py）は
    # 元ビルダーの cells をそのまま連結したもの。Secret 参照は元ビルダー側で
    # 台帳に載っているので、ここでは二重に数えない。
    return "/script/" in path.as_posix()


def test_secrets_match_ledger():
    seen = {}
    for path in (ROOT / "kaggle").rglob("*.py"):
        if "archive" in path.parts or _is_generated(path):
            continue
        names = _names(path)
        if names:
            seen[path.relative_to(ROOT).as_posix()] = names
    # 梱包層（テンプレート内の Secret 参照）も台帳に載せる
    for path in (ROOT / "kbuild").rglob("*.py"):
        if "archive" in path.parts:
            continue
        names = _names(path)
        if names:
            seen[path.relative_to(ROOT).as_posix()] = names
    # `runtime/` は**生成コードへ埋め込まれる真実源**。ここを外すと、
    # セルから実装を runtime へ移した瞬間に Secret の参照が台帳から消える
    # （2026-11-01 に `_gist_status` を runtime/gist.py へ移した）。
    for path in (ROOT / "runtime").rglob("*.py"):
        names = _names(path)
        if names:
            seen[path.relative_to(ROOT).as_posix()] = names

    assert seen == {k: set(v) for k, v in secrets.REQUIRED.items()}, (
        "kbuild/secrets.py を更新してください（台帳と参照が不一致）")

    known = set().union(*secrets.REQUIRED.values())
    assert set().union(*seen.values()) <= known
