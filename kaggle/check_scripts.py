#!/usr/bin/env python3
"""コミット済みの `main.py` がビルダーの出力と一致するか確認する。

`check_notebooks.py` の script 版。ビルダーはリポジトリ内に直接書き込むので、
作業ツリーを汚さないよう一時ディレクトリにコピーしてから再生成し、元の
ファイルとバイト比較する（写す・走らせる・比べるは `kaggle/_regen.py` が持つ）。

`kbuild/spec.py` から対象を引くので、script カーネルを増やしても
このファイルの一覧を更新する必要はない（ビルダーを持つものだけが対象）。

    python3 kaggle/check_scripts.py        # 不一致があれば exit 1
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


def targets():
    """(ビルダー, env, [(出力ディレクトリ, カーネル)]) を台帳から作る。

    `smoke_out` を持つカーネルは、同じビルダーを `KC_SMOKE=1` でもう一度
    走らせ、別ディレクトリの生成物も突き合わせる。通常版を smoke で
    上書きしていないことと、smoke 版の再生成漏れを同時に見る。
    """
    flat = []
    for k in spec.active():
        if k.kind != "script" or not k.builder:
            continue
        flat.append((k.builder, dict(k.env), k.out, k))
        if k.smoke_out:
            flat.append((k.builder, {**dict(k.env), "KC_SMOKE": "1"},
                         k.smoke_out, k))
    # 同じビルダーが複数の出力を出すので、ビルダー単位で 1 回だけ走らせる
    seen, plan = set(), []
    for builder, env, _, _ in flat:
        key = (builder, tuple(sorted(env.items())))
        if key in seen:
            continue
        seen.add(key)
        plan.append((builder, env,
                     [(out, k) for b, e, out, k in flat
                      if (b, tuple(sorted(e.items()))) == key]))
    return plan


def main():
    plan = targets()
    if not plan:
        print("script カーネル（ビルダー持ち）が無い")
        return 0
    failed = []
    with tempfile.TemporaryDirectory(prefix="kc-pycheck-") as tmp:
        work = _regen.stage(tmp)
        for builder, extra, items in plan:
            tag = _regen.label(builder, extra)
            r = _regen.run(work, builder, extra)
            if r.returncode != 0:
                failed.append(tag)
                print(f"❌ {tag}: ビルダーが失敗\n{r.stderr[-800:]}")
                continue
            for out_dir, k in items:
                for name in (k.code_file, k.metadata_file):
                    rel = f"{out_dir}/{name}"
                    state, detail = _regen.diff(work, rel)
                    if _regen.report(rel, state, detail, tag):
                        failed.append(tag)
    if failed:
        print(f"\n不一致 {len(set(failed))} 件")
        return 1
    print("\n全 script がビルダー出力と一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
