#!/usr/bin/env python3
"""コミット済みの生成物がビルダーの出力と一致するか確認する。

notebook（`.ipynb`）と script（`main.py`）を**同じ 1 本**で見る。以前は
`check_notebooks.py` と `check_scripts.py` に分かれていて、差分は `kind` の
判定と文言だけだった（片方だけ直すと食い違う）。

ビルダーはリポジトリ内に直接書き込むので、作業ツリーを汚さないよう一時
ディレクトリにコピーしてから再生成し、元のファイルとバイト比較する
（写す・走らせる・比べる・報告するは `kaggle/_regen.py` が持つ）。

対象は**台帳（`kbuild/spec.py`）から導出**する。生成物のパスは
「台帳の out + code_file」で決まり、smoke 版は `smoke_out` に別名で出る
（`KC_SMOKE=1` を足す）。一覧を手書きすると、新カーネルを足したときに
**検査されないまま commit できた**（生成し忘れを検出するためのチェックが、
そのカーネルだけ素通りする）。

    python3 kaggle/check_generated.py        # 不一致があれば exit 1
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


def _entries(k):
    """(env, 出力ディレクトリ) を列挙する。smoke 版は env が 1 つ違う。"""
    out = [(dict(k.env), k.out)]
    if k.smoke_out:
        out.append(({**dict(k.env), "KC_SMOKE": "1"}, k.smoke_out))
    return out


def targets():
    """(ビルダー, env, [リポジトリ相対の生成物, ...]) を台帳から作る。

    同じビルダーが複数の生成物を出すので、ビルダー単位で 1 回だけ走らせる。
    `metadata_file` も比較する（`code_file` だけ見ていると、台帳と
    `dataset_sources` がずれたまま push される）。
    """
    flat = []
    for k in spec.active():
        if not k.builder:
            continue
        for env, out in _entries(k):
            for name in (k.code_file, k.metadata_file):
                flat.append((k.builder, env, f"{out}/{name}"))

    # 出力先が重複していると、後から走ったビルダーが前の生成物を上書きして
    # 「一致」に見えてしまう。台帳側の設定ミスを先に潰す。
    rels = [rel for _, _, rel in flat]
    dup = sorted({r for r in rels if rels.count(r) > 1})
    if dup:
        raise SystemExit("❌ 生成物のパスが重複: " + ", ".join(dup))

    seen, plan = set(), []
    for builder, env, _ in flat:
        key = (builder, tuple(sorted(env.items())))
        if key in seen:
            continue
        seen.add(key)
        plan.append((builder, env,
                     [rel for b, e, rel in flat
                      if (b, tuple(sorted(e.items()))) == key]))
    return plan


def main():
    plan = targets()
    if not plan:
        print("ビルダーを持つカーネルが無い")
        return 0
    failed = []
    with tempfile.TemporaryDirectory(prefix="kc-gencheck-") as tmp:
        # コミットする生成物は config.env（既定値）で作ったものが基準。
        # config.local.env の個人値（アカウント名など）は焼き込まない前提なので、
        # ここでは写さない。
        if (ROOT / "config.local.env").is_file():
            print("ℹ config.local.env は無視して既定値で比較します"
                  "（個人値で再生成した生成物は不一致になります）\n")
        work = _regen.stage(tmp)
        for builder, extra, rels in plan:
            tag = _regen.label(builder, extra)
            r = _regen.run(work, builder, extra)
            if r.returncode != 0:
                failed.append(tag)
                print(f"❌ {tag}: ビルダーが失敗\n{r.stderr[-800:]}")
                continue
            for rel in rels:
                state, detail = _regen.diff(work, rel)
                if _regen.report(rel, state, detail, tag):
                    failed.append(tag)
    if failed:
        print(f"\n不一致 {len(set(failed))} 件（{len(failed)} ファイル）")
        return 1
    print("\n全生成物がビルダー出力と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
