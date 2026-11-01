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


def for_kernel(kernel, *, code_file=None, kernel_type=None, gpu=None,
               datasets=None, title=None, internet=True, private=True) -> dict:
    """台帳（`kbuild/spec.py` の `Kernel`）から metadata を組み立てる。

    ビルダーが id / title / kind / gpu / datasets を手書きすると、台帳と
    ずれても**誰も気づけない**。`cli build` は台帳の env を渡すだけで、出力は
    ビルダーが書いた値になるため。ここを通せば台帳が唯一の出どころになる。

    `spec` を import しない（循環を避ける）ので、`Kernel` と同じ属性
    （`id` / `title` / `code_file` / `kind` / `gpu` / `datasets`）を持つ
    オブジェクトなら何でもよい。`code_file` だけは script 版で `main.py` に
    なるなど台帳と違うことがあるので、明示で上書きできる。
    """
    return base(
        id=kernel.id,
        title=title or kernel.title,
        code_file=code_file or kernel.code_file,
        kernel_type=kernel_type or kernel.kind,
        gpu=kernel.gpu if gpu is None else gpu,
        datasets=kernel.datasets if datasets is None else datasets,
        internet=internet,
        private=private,
    )


def write(path, meta: dict, newline: bool = False) -> None:
    validate(meta)
    text = json.dumps(meta, indent=2, ensure_ascii=False)
    if newline:
        text += "\n"
    pathlib.Path(path).write_text(text, encoding="utf-8")
