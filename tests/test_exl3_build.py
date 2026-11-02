#!/usr/bin/env python3
"""build_exl3_serve.py の生成物チェック（CPU のみ・Kaggle 不要）。

  - 3 モードが生成でき、再生成が決定的（sha256 一致）
  - API パスワードが result / ログに入らない（P0）
  - Gist 配布のコードと既定値が埋まっている（P0）
  - 認証有効のプロキシを起動している（--no-quota を渡さない）
  - runtime の共通層が埋め込まれている
"""
import ast
import base64
import hashlib
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
import build_exl3_serve as bk  # noqa: E402

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


def test_exl3_build():
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
            check("def publish_url" in body, f"{mode}: URL 配布が無い")
            check("KAGGLE_OWNER" in body, f"{mode}: ビーコンの owner が無い")
            check("GIST" not in body, f"{mode}: Gist が残っている")
            check((out / "kernel-metadata.json").is_file(),
                  f"{mode}: metadata が無い")

            # runtime/ の共通層を埋め込んでいる（exl3 側に実装を残さない）
            for fn in ("kc_detect_gpu", "kc_start_proxy", "kc_wait_url",
                       "kc_wait_http", "kc_read_conf", "kc_write_conf",
                       "kc_stop_via_proxy", "kc_keepalive_expired",
                       "kc_cloudflared_url"):
                check(f"def {fn}" in body, f"{mode}: runtime の {fn} が無い")
            # 起動待ちは無言になる。`--no-progress` で進捗を切り、chat_server は
            # 完走時にしか 1 行出さないので、観測が外れると「遅い」と
            # 「固まった」を区別できない（実測 2026-11-01: 1800 秒で 0 バイト）。
            check("watch, watch_stop = kc_watch(p.pid)" in body,
                  f"{mode}: 起動待ちの観測（kc_watch）が無い")
            check("watch_stop()" in body,
                  f"{mode}: 観測を止めていない（起動後も 30 秒ごとに吐き続ける）")
            check('data["load_watch"] = watch' in body,
                  f"{mode}: 観測結果を result.json に入れていない")
            check("releases/latest" not in body,
                  f"{mode}: cloudflared の URL が latest のまま")
            proxy_src = base64.b64decode(bk.PROXY_B64).decode()
            check("def kc_fetch_quota" in proxy_src,
                  f"{mode}: 埋め込み proxy に quota 実装が無い")
            check("kc_start_proxy(" in body, f"{mode}: proxy 起動が共通層に寄っていない")
            check("kc_stop_via_proxy(" in body, f"{mode}: /stop が共通層に寄っていない")
            check("kc_write_conf(" in body, f"{mode}: conf 書き込みが共通層に寄っていない")
            ast.parse(body)  # テンプレート置換後の構文を CPU で検証する

            again = root / (mode + "-again")
            bk.write(mode, again, meta, minutes)
            check(digest(out / "main.py") == digest(again / "main.py"),
                  f"{mode}: 再生成が決定的でない")


def test_exl3_smoke_variant():
    """--smoke は serve を「疎通 → 終了」に差し替える（GPU 時間の節約）。"""
    with tempfile.TemporaryDirectory(prefix="kc-exl3smoke-") as tmp:
        out = pathlib.Path(tmp) / "serve"
        bk.write("exl3", out, bk.METADATA_REAL, 5, smoke=True)
        body = (out / "main.py").read_text()
        check("SMOKE = True" in body, "smoke: SMOKE フラグが立っていない")
        check("--stop-pid" in body, "smoke: /stop がエンジンを落とさない")
        check("if SMOKE:" in body, "smoke: 分岐が無い")
        # smoke 分岐は tunnel より前にある
        head = body.split("def main():")[1]
        check(head.index("if SMOKE:") < head.index("start_tunnel()"),
              "smoke: tunnel より後に置かれている")
        ast.parse(body)
