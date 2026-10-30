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
GENERATED = {
    ROOT / "kaggle" / "exl3-serve" / d for d in ("kernel", "dryrun", "keepalive")
}


def _names(path):
    src = path.read_text(encoding="utf-8")
    return {m.group(1) for pat in (GET_RE, SECRET_RE) for m in pat.finditer(src)}


def _is_generated(path):
    return any(str(path).startswith(str(d)) for d in GENERATED)


def test_secrets_match_ledger():
    seen = {}
    for path in (ROOT / "kaggle").rglob("*.py"):
        if "archive" in path.parts or _is_generated(path):
            continue
        names = _names(path)
        if names:
            seen[path.relative_to(ROOT).as_posix()] = names

    assert seen == {k: set(v) for k, v in secrets.REQUIRED.items()}, (
        "kbuild/secrets.py を更新してください（台帳と参照が不一致）")

    known = set().union(*secrets.REQUIRED.values())
    assert set().union(*seen.values()) <= known
