"""script カーネル（main.py + kernel-metadata.json）の書き出し。"""
import pathlib
import textwrap

from . import metadata


def wrap_cell(path, header: str, tail: str = "", globals_=()) -> str:
    """セル（トップレベルコード）を関数本体へ機械的に包む。

    notebook のセルは「上から順に実行されるコード」なので、関数化するには
    インデントを付けるだけでよい。中身を書き換えないため、notebook 版と
    script 版で挙動が食い違いにくい。

    `tail`   … 戻り値を確定する行（`launch()` は proc を返す必要がある）
    `globals_` … セルがモジュール変数を更新する場合の名前（`global X` を足す）
    """
    body = pathlib.Path(path).read_text(encoding="utf-8").rstrip()
    head = header
    if globals_:
        head += "    global " + ", ".join(globals_) + "\n"
    return head + textwrap.indent(body, "    ") + tail


def _cell_parts(c):
    """セルを (kind, source) に正規化する。

    ビルダーは 2 通りの形でセルを作る:
      ("code", "...")                                     … pack_nb.write 系
      {"cell_type": "code", "source": ["...", "..."]}     … pack_nb.write_cells 系
    """
    if isinstance(c, dict):
        src = c.get("source") or ""
        if isinstance(src, list):
            src = "".join(src)
        return c.get("cell_type", "code"), src
    return c[0], c[1]


def from_cells(cells, header: str = "", footer: str = "") -> str:
    """notebook のセル列を 1 ファイルの main.py へ**意味を変えずに**連結する。

    セルは「上から順に実行されるコード」なので、順番を保ったまま並べれば
    同じように動く。markdown セルは節コメントにして、diff で構造が読める
    ようにする。IPython マジック（`!cmd` / `%cmd`）は残すと SyntaxError に
    なるので、ここで検出して落とす（黙って壊れた main.py を出さない）。
    """
    out = [header.rstrip()] if header else []
    for i, c in enumerate(cells):
        kind, src = _cell_parts(c)
        body = src.rstrip("\n")
        if not body.strip():
            continue
        if kind == "markdown":
            # 見出し行だけ拾って節コメントにする（本文は落とす）
            heads = [ln.strip() for ln in body.splitlines()
                     if ln.strip().startswith("#")]
            title = heads[0].lstrip("# ").strip() if heads else ""
            out.append(f"# ── {title} " + "─" * max(0, 44 - len(title))
                       if title else f"# ── cell {i} " + "─" * 40)
            continue
        for ln in body.splitlines():
            if ln.lstrip().startswith(("!", "%")):
                raise ValueError(
                    f"セル {i} に IPython マジックがある（script 化できない）: "
                    f"{ln.strip()[:60]}")
        out.append(body)
    text = "\n\n\n".join(out)
    return text.rstrip() + "\n" + (footer.rstrip() + "\n" if footer else "")


def write(out_dir, body: str, meta: dict, newline_meta: bool = True) -> None:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main.py").write_text(body, encoding="utf-8")
    metadata.write(out_dir / "kernel-metadata.json", meta, newline=newline_meta)
    print(f"wrote {out_dir}/main.py ({len(body):,} B)")
