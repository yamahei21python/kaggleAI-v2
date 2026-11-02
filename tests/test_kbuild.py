"""kbuild の梱包層の自己テスト（CPU のみ）。"""
import json
import pathlib
import tempfile

from kbuild import embed, metadata, pack_nb, pack_script, placeholders


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_kbuild():
    # title → slug 検査
    check(metadata.slugify("llama-server ngrok")
          == "llama-server-ngrok", "slugify")
    meta = metadata.base(
        id="owner/llama-server-ngrok",
        title="llama-server ngrok",
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


def test_placeholders_guard():
    """置換漏れ（`__NAME__` の残り）を書き出し口で落とす。

    2026-11-02 まで、ビルダーの `str.replace` が漏れても生成物は静かに
    通っていた。`KAGGLE_OWNER = "__KAGGLE_OWNER__"` のような残骸は
    ビルドも構文検査も通り、GPU を掴んだあとに初めて壊れる。
    """
    # 拾うのは大文字始まりの dunder だけ（`__all__` は対象外）
    check(placeholders.leftovers('X = "__MODEL__"') == ["MODEL"], "検出")
    check(placeholders.leftovers("__all__ = []\n__init__ = 1") == [],
          "小文字の dunder を拾っている")
    check(placeholders.leftovers("") == [], "空文字")

    # どのトークンが残ったかを名前で言う
    try:
        placeholders.check('a = "__A__"\nb = "__B__"', "どこか")
    except ValueError as e:
        msg = str(e)
        check("__A__" in msg and "__B__" in msg, f"トークン名が無い: {msg}")
        check("どこか" in msg, f"場所が無い: {msg}")
        check("置換漏れ" in msg, f"原因が読めない: {msg}")
    else:
        raise AssertionError("残骸を検出できない")

    # notebook: セル番号つきで落ちる
    try:
        pack_nb.notebook([("code", "ok = 1\n"),
                          ("code", 'SRC = "__DATASET__"\n')])
    except ValueError as e:
        check("セル 1" in str(e), f"セル番号が無い: {e}")
    else:
        raise AssertionError("notebook の残骸を検出できない")

    # script: 落ちたときは main.py を書かない（半端な生成物を残さない）
    meta = metadata.base(
        id="owner/x", title="x", code_file="main.py",
        kernel_type="script", gpu=False)
    with tempfile.TemporaryDirectory(prefix="kbuild-ph-") as tmp:
        out = pathlib.Path(tmp) / "kernel"
        try:
            pack_script.write(out, 'N = "__SERVED__"\n', meta)
        except ValueError:
            pass
        else:
            raise AssertionError("script の残骸を検出できない")
        check(not (out / "main.py").exists(), "落ちたのに main.py を書いている")
