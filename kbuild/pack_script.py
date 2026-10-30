"""script カーネル（main.py + kernel-metadata.json）の書き出し。"""
import pathlib

from . import metadata


def write(out_dir, body: str, meta: dict, newline_meta: bool = True) -> None:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main.py").write_text(body, encoding="utf-8")
    metadata.write(out_dir / "kernel-metadata.json", meta, newline=newline_meta)
    print(f"wrote {out_dir}/main.py ({len(body):,} B)")
