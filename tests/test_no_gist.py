"""Gist 経路が戻っていないことを固定する（2026-11-02 に全廃）。

公開 URL の配布はビーコンカーネル（`runtime/beacon.py`）に一本化した。
Gist を消した理由は `notes/url-publishing.md`:

  Kaggle の Secret は**カーネルごとに紐付ける**必要があり、未紐付けの
  `get_secret()` は `ConnectionError`（＝通信障害に見える）を返す。
  カーネルを増やすたびに UI 作業が要り、しかも失敗しても
  「何も起きていない」と区別がつかない。

**コードと設定に戻っていないこと**だけを見る。docs / notes に経緯として
「Gist」と書くのは構わない（消した理由を残すほうが大事）ので、対象は
実行されるソースに限る。
"""
import pathlib
import re

import runtime
from kbuild import spec

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 実行されるソースだけ。生成物（kaggle/**）はビルダー出力なので、
# ビルダー側を見れば足りる（`check_generated` が一致を保証している）。
SOURCES = [
    *sorted((ROOT / "runtime").rglob("*.py")),
    *sorted((ROOT / "kbuild").rglob("*.py")),
    *sorted((ROOT / "cli").rglob("*.py")),
    *sorted((ROOT / "kaggle").glob("build_*.py")),
    *sorted((ROOT / "kaggle" / "serve-cells").glob("*.py")),
    *sorted((ROOT / "kaggle" / "step2" / "cells").glob("*.py")),
    *sorted((ROOT / "kaggle" / "step2-vllm" / "cells").glob("*.py")),
    ROOT / "kaggle" / "check_generated.py",
]

# 機能として効いてしまうものだけを禁じる。「Gist は廃止した」という
# 説明文は通す（そうしないと、なぜ消したかを書けなくなる）。
BANNED = {
    "Gist の API": r"api\.github\.com/gists",
    "Gist の生 URL": r"gist\.githubusercontent\.com",
    "Gist ID の設定": r"GIST_ID",
    "Gist 用のトークン": r"GITHUB_TOKEN",
    "Gist 書き込み関数": r"kc_gist_\w+",
    "Gist のモジュール": r"runtime/gist\.py|\"gist\"\s*:|'gist'\s*:",
}


def test_no_gist_code_is_left():
    hits = []
    for path in SOURCES:
        src = path.read_text(encoding="utf-8")
        for label, pat in BANNED.items():
            for m in re.finditer(pat, src):
                line = src.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(ROOT)}:{line} — {label}")
    assert not hits, "Gist 経路が戻っている:\n" + "\n".join(hits)


def test_gist_module_and_cell_are_gone():
    assert not (ROOT / "runtime" / "gist.py").exists()
    assert not (ROOT / "kaggle" / "serve-cells" / "gist.py").exists()
    assert "gist" not in runtime.MODULES


def test_ledger_has_no_gist_kernel():
    names = [k.name for k in spec.KERNELS]
    assert not [n for n in names if "gist" in n.lower()], names


def test_publish_cell_is_the_only_publisher():
    """配布セルは 1 本だけ（`publish.py`）。名前が増えたら並びも見直す。"""
    cells = sorted(p.name for p in (ROOT / "kaggle" / "serve-cells").glob("*.py"))
    assert "publish.py" in cells, cells
    assert not [c for c in cells if "gist" in c], cells
