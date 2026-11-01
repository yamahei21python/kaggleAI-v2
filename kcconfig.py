"""後方互換シム。実体は `cli.config`。

`kaggle/build_*.py` は `import kcconfig` で config.env を読む。
"""
from cli.config import ROOT, _parse, load  # noqa: F401
