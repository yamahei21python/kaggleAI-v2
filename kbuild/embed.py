"""base64 埋め込み。script カーネルは code_file を 1 つしか持てない。"""
import base64
import pathlib


def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_file(path) -> str:
    return b64_bytes(pathlib.Path(path).read_bytes())
