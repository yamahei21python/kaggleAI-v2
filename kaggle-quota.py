#!/usr/bin/env python3
"""後方互換シム。実体は `cli.quota`（`python3 -m cli quota` でも同じ）。"""
import sys

from cli.quota import main

if __name__ == "__main__":
    sys.exit(main())
