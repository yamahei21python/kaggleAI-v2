#!/usr/bin/env python3
"""コミット済みのノートブックがビルダーの出力と一致するか確認する。

ビルダーはリポジトリ内に直接書き込むので、作業ツリーを汚さないよう
一時ディレクトリにコピーしてから再生成し、元のファイルと比べる
（写す・走らせる・比べるは `kaggle/_regen.py` が持つ）。
auth_proxy.py / llama-chat.html を直したのに ipynb を再生成し忘れると、
Kaggle には古いコードが push される（実際に ngrok 版でずれていた）。

    python3 kaggle/check_notebooks.py        # 不一致があれば exit 1
"""
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import _regen  # noqa: E402
from kbuild import spec  # noqa: E402

# (ビルダー, 追加の環境変数, 生成物) — 生成物はリポジトリ相対。
#
# **手書きしない。** 台帳（`kbuild/spec.py`）から導出する。
# notebook の生成物は「台帳の out + code_file」で決まり、smoke 版は
# `smoke_out` に別名で出る（`KC_SMOKE=1` を足す）。以前はここに 10 行
# 直書きしていて、新カーネルを足すと**検査されないまま commit できた**
# （生成し忘れを検出するためのチェックが、そのカーネルだけ素通りする）。
def _targets():
    out = []
    for k in spec.KERNELS:
        if k.kind != "notebook" or k.status == spec.RETIRED or not k.builder:
            continue
        out.append((f"{k.builder}.py", dict(k.env),
                    f"{k.out}/{k.code_file}"))
        if k.smoke_out:
            out.append((f"{k.builder}.py", {**dict(k.env), "KC_SMOKE": "1"},
                        f"{k.smoke_out}/{k.code_file}"))
    return out


TARGETS = _targets()


def main():
    # 出力先が重複していると、後から走ったビルダーが前の生成物を上書きして
    # 「一致」に見えてしまう。TARGETS の設定ミスを先に潰す。
    outs = [out for _, _, out in TARGETS]
    dup = sorted({o for o in outs if outs.count(o) > 1})
    if dup:
        print("❌ TARGETS の出力先が重複: " + ", ".join(dup))
        return 1
    failed = []
    with tempfile.TemporaryDirectory(prefix="kc-nbcheck-") as tmp:
        # コミットするノートは config.env（既定値）で生成したものが基準。
        # config.local.env の個人値はノートに焼き込まず、Kaggle 側は Secrets
        # （GIST_ID）で上書きする前提なので、ここでは写さない。
        if (ROOT / "config.local.env").is_file():
            print("ℹ config.local.env は無視して既定値で比較します"
                  "（個人値で再生成したノートは不一致になります）\n")
        work = _regen.stage(tmp)
        for builder, extra, out in TARGETS:
            tag = _regen.label(builder, extra)
            r = _regen.run(work, builder, extra)
            if r.returncode != 0:
                failed.append(tag)
                print(f"❌ {tag}: ビルダーが失敗\n{r.stderr[-800:]}")
                continue
            state, detail = _regen.diff(work, out)
            if _regen.report(out, state, detail, tag):
                failed.append(tag)
    if failed:
        print(f"\n不一致 {len(failed)} 件")
        return 1
    print("\n全ノートブックがビルダー出力と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
