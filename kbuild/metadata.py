"""kernel-metadata.json の生成と title↔slug 検査。

Kaggle は title から slug を作る。id の slug と一致しないと push が
409 Conflict になるため、書き出し前に必ず検査する。
"""
import json
import pathlib
import re


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return slug.strip("-")


def validate(meta: dict) -> None:
    kernel_id = meta.get("id", "")
    slug = kernel_id.rsplit("/", 1)[-1]
    title_slug = slugify(meta.get("title", ""))
    if title_slug != slug:
        raise ValueError(
            f"title {meta.get('title')!r} は slug {title_slug!r} になり、"
            f"id {kernel_id!r} と一致しない")


def base(
    *,
    id: str,
    title: str,
    code_file: str,
    kernel_type: str,
    gpu: bool,
    datasets=(),
    internet: bool = True,
    private: bool = True,
) -> dict:
    return {
        "id": id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": kernel_type,
        "is_private": private,
        "enable_gpu": gpu,
        "enable_tpu": False,
        "enable_internet": internet,
        "keywords": [],
        "dataset_sources": list(datasets),
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }


def write(path, meta: dict, newline: bool = False) -> None:
    validate(meta)
    text = json.dumps(meta, indent=2, ensure_ascii=False)
    if newline:
        text += "\n"
    pathlib.Path(path).write_text(text, encoding="utf-8")
