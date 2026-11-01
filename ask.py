#!/usr/bin/env python3
"""後方互換シム。実体は `cli.ask`（`python3 -m cli ask` でも同じ）。"""
import sys

from cli.ask import main

if __name__ == "__main__":
    sys.exit(main())
