#!/usr/bin/env python3
"""build_kernel.py の生成物チェック（CPU のみ・Kaggle 不要）。

  - 3 モードが生成でき、再生成が決定的（sha256 一致）
  - API パスワードが result / ログに入らない（P0）
  - Gist 配布のコードと既定値が埋まっている（P0）
  - 認証有効のプロキシを起動している（--no-quota を渡さない）
"""
import hashlib
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_kernel as bk  # noqa: E402

MODES = [
    ("exl3", bk.METADATA_REAL, 60),
    ("mock", bk.METADATA_DRYRUN, 5),
    ("keepalive", bk.METADATA_KEEPALIVE, 2),
]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory(prefix="kc-exl3build-") as tmp:
        root = pathlib.Path(tmp)
        for mode, meta, minutes in MODES:
            out = root / mode
            bk.write(mode, out, meta, minutes)
            body = (out / "main.py").read_text()
            check(f'MODE = "{mode}"' in body, f"{mode}: MODE")
            check("api_pass" not in body,
                  f"{mode}: 生成物に api_pass が含まれる")
            check('"API_PASS": pass_' in body,
                  f"{mode}: プロキシへ API_PASS を渡していない")
            check("--no-quota" not in body,
                  f"{mode}: --no-quota が残っている")
            check("def publish_url" in body, f"{mode}: Gist 配布が無い")
            check("GIST_ID_DEFAULT" in body, f"{mode}: Gist 既定値が無い")
            check((out / "kernel-metadata.json").is_file(),
                  f"{mode}: metadata が無い")

            again = root / (mode + "-again")
            bk.write(mode, again, meta, minutes)
            check(digest(out / "main.py") == digest(again / "main.py"),
                  f"{mode}: 再生成が決定的でない")

    print("OK: exl3 build selftest")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
