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
    relay = _read("run-kaggle-chat.command")
    assert f'KC_RELAY_PORT:-${{CFG_RELAY_PORT:-{constants.RELAY_PORT}}}' in relay


def test_ui_default_endpoint():
    assert f"127.0.0.1:{constants.CHAT_PORT}/v1" in _read("llama-chat.html")
