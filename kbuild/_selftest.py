#!/usr/bin/env python3
"""kbuild の梱包層の自己テスト（CPU のみ）。"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from kbuild import embed, metadata, pack_nb, pack_script  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    # title → slug 検査
    check(metadata.slugify("Step2 direct llama-server ngrok")
          == "step2-direct-llama-server-ngrok", "slugify")
    meta = metadata.base(
        id="owner/step2-direct-llama-server-ngrok",
        title="Step2 direct llama-server ngrok",
        code_file="x.ipynb", kernel_type="notebook", gpu=True)
    metadata.validate(meta)
    try:
        metadata.validate({**meta, "title": "mismatch"})
    except ValueError:
        pass
    else:
        raise AssertionError("title 不一致を検出できない")

    with tempfile.TemporaryDirectory(prefix="kbuild-selftest-") as tmp:
        tmp = pathlib.Path(tmp)

        # notebook: cell id の有無と JSON の形
        cells = [("markdown", "a\n"), ("code", "print(1)\n")]
        plain = pack_nb.notebook(cells)
        check("id" not in plain["cells"][0], "cell id が既定で付く")
        check(plain["cells"][1]["outputs"] == [], "code cell の outputs")
        with_ids = pack_nb.notebook(cells, cell_ids=True)
        check(with_ids["cells"][0]["id"] == "cell-00", "cell id 採番")

        nb_path = tmp / "a.ipynb"
        pack_nb.write(nb_path, cells)
        raw = nb_path.read_text(encoding="utf-8")
        check(json.loads(raw)["nbformat_minor"] == 5, "nbformat")
        check(not raw.endswith("\n"), "notebook に末尾改行を足している")

        # metadata: 末尾改行の切り替え
        meta_path = tmp / "kernel-metadata.json"
        metadata.write(meta_path, meta)
        check(not meta_path.read_text(encoding="utf-8").endswith("\n"),
              "metadata の既定に末尾改行がある")
        metadata.write(meta_path, meta, newline=True)
        check(meta_path.read_text(encoding="utf-8").endswith("\n"),
              "metadata の newline=True が効いていない")

        # script: main.py と metadata の同時生成
        pack_script.write(tmp / "kernel", "body\n", meta)
        check((tmp / "kernel" / "main.py").read_text(encoding="utf-8") == "body\n",
              "main.py")
        check((tmp / "kernel" / "kernel-metadata.json").is_file(), "metadata")

        # base64: 往復
        blob = tmp / "blob.bin"
        blob.write_bytes(bytes(range(256)))
        import base64
        check(base64.b64decode(embed.b64_file(blob)) == bytes(range(256)),
              "base64 往復")

    print("OK: kbuild selftest")


if __name__ == "__main__":
    main()
