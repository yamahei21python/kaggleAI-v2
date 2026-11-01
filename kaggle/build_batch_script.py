#!/usr/bin/env python3
"""配信しないカーネル（データ準備 / 検証 / ベンチ）を **script 形式**で生成する。

    KC_BATCH=build_step1 KC_BATCH_OUT=kaggle/step1/script \
        python3 kaggle/build_batch_script.py

`cli build` は台帳（`kbuild/spec.py`）の env を渡して呼ぶので、通常は
ビルダーを直接叩かない。

notebook 版はそのまま残す。こちらは**同じ `cells` をそのまま 1 ファイルへ
連結する**だけの追加生成物で、セルの中身には手を触れない（＝意味が変わらない）。
骨組み（`serve_script`）は配信カーネル用なので、ここでは使わない。

id / title / GPU / Dataset は台帳から引く（二重管理しない）。
"""
import importlib
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from kbuild import metadata, pack_script, spec  # noqa: E402

SRC = os.environ.get("KC_BATCH", "").strip()
OUT_REL = os.environ.get("KC_BATCH_OUT", "").strip()
if not SRC or not OUT_REL:
    raise SystemExit("KC_BATCH / KC_BATCH_OUT を環境変数で渡す")

kernel = next((k for k in spec.KERNELS if k.out == OUT_REL), None)
if kernel is None:
    raise SystemExit(f"台帳に {OUT_REL} が無い（kbuild/spec.py を確認）")

mod = importlib.import_module(SRC)
cells = getattr(mod, "cells", None)
if not cells:
    raise SystemExit(f"{SRC} が cells を持たない（ビルダーを確認）")

HEADER = (
    '#!/usr/bin/env python3\n'
    f'"""{(kernel.title or kernel.name)}（自動生成。直接編集しない）。\n\n'
    f"生成: kaggle/build_batch_script.py ← kaggle/{SRC}.py のセル列\n"
    '"""\n'
)

body = pack_script.from_cells(cells, header=HEADER)

META = metadata.base(
    id=kernel.id,
    title=kernel.title,
    code_file="main.py",
    kernel_type="script",
    gpu=kernel.gpu,
    datasets=kernel.datasets,
)

if __name__ == "__main__":
    pack_script.write(ROOT / OUT_REL, body, META)
