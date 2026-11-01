"""ポート等の固定値が各所でずれていないか検査する。

カーネル内で単体実行されるファイル（auth_proxy.py / chat_server.py）と
ランチャーは import できないため、リテラルと runtime/constants.py を突き合わせる。
"""
import pathlib
import re

from runtime import constants

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_auth_proxy_default_port():
    src = _read("kaggle/step2/auth_proxy.py")
    m = re.search(r'--port",\s*type=int,\s*default=(\d+)', src)
    assert m and int(m.group(1)) == constants.PROXY_PORT


def test_chat_server_default_port():
    src = _read("kaggle/exl3-serve/chat_server.py")
    assert f"port={constants.CHAT_PORT}" in src
    assert f'default={constants.CHAT_PORT}' in src


def test_builders_use_constants():
    """生成物の PROXY_PORT は constants と一致する。"""
    for path in sorted((ROOT / "kaggle").glob("build_*.py")):
        for m in re.finditer(r"^PROXY_PORT = (\d+)", path.read_text(encoding="utf-8"),
                             re.MULTILINE):
            assert int(m.group(1)) == constants.PROXY_PORT, path.name
        for m in re.finditer(r"^PORT = (\d+)", path.read_text(encoding="utf-8"),
                             re.MULTILINE):
            assert int(m.group(1)) in (constants.KEEPALIVE_TEST_PORT,
                                       constants.OLLAMA_PORT,
                                       constants.CHAT_PORT), path.name


def test_config_env_ports():
    cfg = _read("config.env")
    values = dict(line.split("=", 1) for line in cfg.splitlines()
                  if "=" in line and not line.startswith("#"))
    assert int(values["UZU_PORT"]) == constants.CHAT_PORT
    assert int(values["UI_PORT"]) == constants.UI_PORT
    assert int(values["RELAY_PORT"]) == constants.RELAY_PORT


def test_launchers_default_ports():
    llama = _read("run-llama-server.command")
    assert f'LLAMA_PORT:-{constants.CHAT_PORT}' in llama
    assert f'LLAMA_UI_PORT:-{constants.UI_PORT}' in llama
    # 中継を持つのは run-ai.command（run-kaggle-chat は ngrok の警告ページ
    # 回避のためだけに中継を立てていたので、2026-11-01 に落とした）。
    relay = _read("run-ai.command")
    assert f'KC_RELAY_PORT:-${{CFG_RELAY_PORT:-{constants.RELAY_PORT}}}' in relay


def test_ui_default_endpoint():
    assert f"127.0.0.1:{constants.CHAT_PORT}/v1" in _read("llama-chat.html")


def test_launchers_have_no_ngrok_left():
    """ランチャーと config.env に ngrok の配線が残っていないこと。

    ngrok は 2026-11-01 に廃止した（`notes/url-publishing.md`）。固定ドメインを
    前提にした分岐が残っていると、**存在しない URL を開こうとする**か、
    retired にしたカーネルを push しようとする。どちらも黙って失敗するので、
    名前そのものを禁止する。URL はビーコン（`kc_beacon_url`）から読む。
    """
    for name in ("run-kaggle-chat.command", "run-ai.command"):
        src = _read(name)
        for bad in ("CFG_NGROK_DOMAIN", "CFG_KERNEL_SLUG_NGROK",
                    "step2-ngrok", "KC_TUNNEL:-ngrok"):
            assert bad not in src, f"{name} に {bad} が残っている"
    assert "NGROK" not in _read("config.env")
