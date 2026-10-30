#!/usr/bin/env python3
"""build_kernel.py の生成物チェック（CPU のみ・Kaggle 不要）。

  - 3 モードが生成でき、再生成が決定的（sha256 一致）
  - API パスワードが result / ログに入らない（P0）
  - Gist 配布のコードと既定値が埋まっている（P0）
  - 認証有効のプロキシを起動している（--no-quota を渡さない）
"""
import ast
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

            # runtime/ の共通層を埋め込んでいる（exl3 側に実装を残さない）
            for fn in ("kc_detect_gpu", "kc_start_proxy", "kc_wait_url",
                       "kc_wait_http", "kc_read_conf", "kc_write_conf",
                       "kc_stop_via_proxy", "kc_keepalive_deadline"):
                check(f"def {fn}" in body, f"{mode}: runtime の {fn} が無い")
            check("kc_start_proxy(" in body, f"{mode}: proxy 起動が共通層に寄っていない")
            check("kc_stop_via_proxy(" in body, f"{mode}: /stop が共通層に寄っていない")
            check("kc_write_conf(" in body, f"{mode}: conf 書き込みが共通層に寄っていない")
            ast.parse(body)  # テンプレート置換後の構文を CPU で検証する

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
