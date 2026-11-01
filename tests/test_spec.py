"""台帳（kbuild/spec.py）と生成物メタデータの一致を検査する。

spec を唯一の真実源にするため、生成済みの kernel-metadata.json と
突き合わせる。ズレたらどちらかを直す。
"""
import json

import pytest

from kbuild import spec


def test_ids_and_names_unique():
    names = [k.name for k in spec.all_kernels()]
    assert len(names) == len(set(names))
    ids = [k.id for k in spec.all_kernels()]
    assert len(ids) == len(set(ids))


def test_title_matches_slug():
    """既存 id は title から解決されない（作成済みのため）ので、形だけ確認する。"""
    for k in spec.all_kernels():
        assert k.title.strip()
        assert k.id.startswith("yamahei21python/")
        assert k.name == k.name.lower()


# 移行前から存在するカーネル。title を変えると slug が変わり、既存 id と
# 衝突して push が 409 になるため、そのままにしてある。
LEGACY_TITLE_MISMATCH = {
    "step1-orcarouter-qwen38-27b-dl",       # "Qwen3.8" が qwen3-8 になる
    "capture-ollama-patch-v2",              # 退役
}


def test_new_kernel_title_slugifies_to_id():
    """新しく作るカーネルは title がそのまま slug になる必要がある。

    Kaggle は title から slug を作るので、ずれていると **新規作成が 409 Conflict**。
    2026-10-30 に `Smoke: orcarouter Qwen3.8-27B on T4x2` がこれで作れなかった。
    """
    from kbuild import metadata
    bad = [f"{k.name}: {k.title!r} -> {metadata.slugify(k.title)}"
           for k in spec.all_kernels()
           if k.status != spec.RETIRED
           and k.name not in LEGACY_TITLE_MISMATCH
           and metadata.slugify(k.title) != k.name]
    assert not bad, "title と slug がずれている:\n" + "\n".join(bad)


def test_title_fits_kaggle_limit():
    """title は 50 文字以内（超えると SaveKernel が 400 Bad Request）。

    slug は title から作られるので、長い id はそもそも作れない。
    2026-10-30 に 53 文字の `bench-mtp-spec-decoding-…-script` がこれで作れず、
    `bench-mtp-orcarouter-qwen3-8-27b-script` に短縮した。
    """
    too_long = [f"{k.name}: {len(k.title)} 文字" for k in spec.all_kernels()
                if len(k.title) > 50]
    assert not too_long, "title が 50 文字を超えている:\n" + "\n".join(too_long)


def test_builder_exists():
    for k in spec.all_kernels():
        if not k.builder:
            continue
        path = spec.ROOT / "kaggle" / f"{k.builder}.py"
        assert path.is_file(), f"{k.id}: {path} が無い"


def test_output_dir_exists():
    for k in spec.all_kernels():
        if k.status == spec.RETIRED:
            continue
        assert k.dir.is_dir(), f"{k.id}: {k.dir} が無い"


@pytest.mark.parametrize("kernel", spec.all_kernels(), ids=lambda k: k.name)
def test_metadata_matches_spec(kernel):
    path = kernel.metadata_path
    assert path.is_file(), f"メタデータが無い: {path}"
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["id"] == kernel.id
    assert meta["title"] == kernel.title
    assert meta["kernel_type"] == kernel.kind
    assert meta["enable_gpu"] is kernel.gpu
    assert meta["code_file"] == kernel.code_file
    # 退役したカーネルは台帳から datasets を落としてある（もう push しない
    # ので、二度と使わない値を残すと「まだ動く」と読めてしまう）。メタデータ
    # 側は履歴としてそのまま残すため、ここだけ比較しない。
    # 2026-11-01 の ngrok 廃止で 6 本がこの形になった。
    if kernel.status != spec.RETIRED:
        assert tuple(meta["dataset_sources"]) == kernel.datasets


def test_smoke_output_is_separate_from_normal():
    """smoke は通常版と別パス（同じだと push --smoke が通常版を上書きする）。"""
    with_smoke = [k for k in spec.all_kernels() if k.smoke_out]
    assert with_smoke, "smoke_out を持つカーネルが無い"
    for k in with_smoke:
        assert k.smoke_dir != k.dir, f"{k.id}: smoke と通常版が同じ出力先"
        assert k.smoke_dir.is_dir(), f"{k.id}: {k.smoke_dir} が無い"
        meta = json.loads(
            (k.smoke_dir / k.metadata_file).read_text(encoding="utf-8"))
        # id は通常版と共有（push 先は同じカーネル）。code_file 名も同じ。
        assert meta["id"] == k.id
        assert meta["code_file"] == k.code_file


def test_smoke_out_belongs_to_smoke_capable_builder():
    """smoke_out を持つカーネルのビルダーは `--smoke` を受ける必要がある。

    受けないビルダーに smoke_out を書くと、push --smoke が通常版を staging
    してしまう（`cli/build.py` の SMOKE_BUILDERS が唯一の判定）。
    """
    from cli import build as build_mod
    for k in spec.all_kernels():
        if not k.smoke_out:
            continue
        assert k.builder in build_mod.SMOKE_BUILDERS, \
            f"{k.id}: {k.builder} は --smoke 非対応"


def test_builders_plan_is_deduped():
    plan = spec.builders()
    keys = [(b, tuple(sorted(env.items()))) for b, env in plan]
    assert len(keys) == len(set(keys))
    assert ("build_step2", {"STEP2_TUNNEL": "cloudflared"}) in plan
    # ngrok は 2026-11-01 に廃止。骨格（builder）は cloudflared 以外を
    # 受け付けないので、計画に混ざっていたら台帳の書き戻し漏れ。
    assert ("build_step2", {"STEP2_TUNNEL": "ngrok"}) not in plan
    assert all(env.get("STEP2_TUNNEL", "cloudflared") == "cloudflared"
               for _b, env in plan)


def test_find_accepts_name_and_id_and_path():
    kernel = spec.find("exl3-serve")
    assert spec.find("yamahei21python/exl3-serve") is kernel
    assert spec.find("kaggle/exl3-serve-kernel/serve") is kernel
    with pytest.raises(KeyError):
        spec.find("nope")
