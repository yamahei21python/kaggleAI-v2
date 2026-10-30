"""notebook の組み立て。nbformat_minor=5 では cell id が必須ではないが、
一部のビルダーは `cell-NN` を付けて生成しているので切り替えられるようにする。
"""
import json
import pathlib


def notebook(cells, language_version: str = "3.10.0", cell_ids: bool = False) -> dict:
    built = []
    for i, (kind, source) in enumerate(cells):
        cell = {"cell_type": kind}
        if cell_ids:
            cell["id"] = f"cell-{i:02d}"
        cell["metadata"] = {}
        cell["source"] = source.splitlines(keepends=True)
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        built.append(cell)
    return {
        "cells": built,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": language_version},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def notebook_from_cells(cells, language_version: str = "3.10.0") -> dict:
    """既に dict で組み立て済みの cells を notebook に包む。"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": language_version},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_cells(path, cells, language_version: str = "3.10.0") -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook_from_cells(cells, language_version), f,
                  ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(cells)} cells")


def write(path, cells, language_version: str = "3.10.0",
          cell_ids: bool = False) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook(cells, language_version, cell_ids), f,
                  ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(cells)} cells")
