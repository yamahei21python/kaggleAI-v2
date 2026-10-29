"""config.env / config.local.env を読む（標準ライブラリのみ）。

形式は KEY=VALUE の行だけ。# 始まりと空行は無視する。
config.local.env（gitignore）の値が config.env より優先される。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def _parse(path):
    out = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def load(root=ROOT):
    root = pathlib.Path(root)
    cfg = _parse(root / "config.env")
    cfg.update(_parse(root / "config.local.env"))
    return cfg
