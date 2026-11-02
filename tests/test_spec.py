"""台帳（kbuild/spec.py）と生成物メタデータの一致を検査する。

spec を唯一の真実源にするため、生成済みの kernel-metadata.json と
突き合わせる。ズレたらどちらかを直す。
"""
import ast
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


def test_builders_take_metadata_from_the_ledger():
    """ビルダーは `metadata.for_kernel` で台帳から引く（手書きしない）。

    2026-11-02 まで 10 本のビルダーが `metadata.base(id=..., title=...,
    gpu=..., datasets=[...])` を手書きしていた。台帳と二重管理になるので、
    片方だけ直しても**生成物を見比べるまで誰も気づけない**。

    例外は 2 本だけで、どちらも「台帳に無いカーネル」を作る:
      - `build_serve_url_beacon.py` … 任意の slug（`KC_BEACON_SLUG`）で掲示板を作る
      - `build_llama_server_script.py` … llama-server 系は台帳に id が無い（退役）
    """
    allowed = {"build_serve_url_beacon.py", "build_llama_server_script.py"}
    offenders = []
    for path in sorted(spec.ROOT.glob("kaggle/build_*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "base"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "metadata"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "台帳を使わず metadata.base を手書きしている: " + ", ".join(offenders))


def test_no_module_imports_the_kcconfig_shim():
    """`kcconfig` シムを import しているモジュールが無いこと。

    `kcconfig.py` は `cli/config.py` への後方互換シムで、ビルダー 9 本が
    `import kcconfig` していた。シムがあると「実体はどちらか」が読むたびに
    必要になるので、ビルダー側を `from cli import config` に寄せてシムを消した。
    戻ってきたら気づけるよう、import の有無だけ見る。

    走査は**リポジトリのコードが入っている場所だけ**。ルート全体を `**/*.py`
    で舐めると `.venv-kaggle-cli/` まで入り、Python 3.10 以降の構文
    （`match`）で `ast.parse` が落ちる（実測: 3.9 で SyntaxError）。
    """
    paths = list(spec.ROOT.glob("*.py"))
    for name in ("kaggle", "kbuild", "runtime", "cli", "tests"):
        paths += (spec.ROOT / name).rglob("*.py")
    offenders = []
    for path in sorted(paths):
        if "archive" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "kcconfig" or n.startswith("kcconfig.")
                   for n in names):
                offenders.append(f"{path.relative_to(spec.ROOT)}:{node.lineno}")
    assert not offenders, "kcconfig シムを import している: " + ", ".join(offenders)


def test_env_preamble_is_empty_without_knobs():
    """焼く既定値が無いときは**空文字**（生成物に 1 バイトも足さない）。"""
    from kbuild.settings_cell import env_preamble
    assert env_preamble(()) == ""
    assert env_preamble({}) == ""
    assert env_preamble((("A", "1"),)) == (
        "import os\nos.environ.setdefault('A', '1')\n\n")


def test_vllm_production_kernels_bake_the_prefetch_knobs():
    """本番の vLLM 2 カーネルが prefetch を焼いていること。

    playground の `prefetch1` で重みロードが 2,877 → 1,186 秒、起動全体が
    3,820 → 2,011 秒になった（`notes/vllm-startup-speedup.md` §23）ので既定に
    した。台帳の `bake_env` を落とすと**静かに遅い既定へ戻る**ので、生成物まで
    見る（notebook は JSON の中に同じ行が入る）。
    """
    for name in ("step2-vllm-qwen38-27b-cloudflared",
                 "step2-vllm-qwen38-27b-cloudflared-script"):
        k = spec.find(name)
        assert dict(k.bake_env) == dict(spec.VLLM_BAKE_ENV), name
        text = (k.dir / k.code_file).read_text(encoding="utf-8")
        for key, val in k.bake_env:
            assert f"os.environ.setdefault({key!r}, {val!r})" in text, name


def test_startup_budget_is_baked_as_a_pair():
    """起動予算は **2 つの env で 1 つのノブ**。片方だけ焼くと効かない。

    `03_vllm_serve.py` の 1 試行の上限は

        _tmo = min(ATTEMPT_TIMEOUT, max(45, STARTUP_BUDGET - 経過))

    なので、`VLLM_STARTUP_BUDGET` だけ 7,000 にしても `ATTEMPT_TIMEOUT` が
    4,200 のままだと 4,200 秒で切られる（2026-11-02 に一度踏んでいる）。

    下限を 6,500 にしてあるのは実測から: 本番 v16 は 62 層中 56 層で
    4,222 秒に達し、完走には 4,650 秒前後必要だった
    （`notes/vllm-startup-speedup.md` §24）。
    """
    d = dict(spec.VLLM_BAKE_ENV)
    assert "VLLM_STARTUP_BUDGET" in d, "予算を焼いていない"
    assert "VLLM_ATTEMPT_TIMEOUT" in d, \
        "片方だけでは効かない（_tmo が ATTEMPT_TIMEOUT で切られる）"
    assert d["VLLM_STARTUP_BUDGET"] == d["VLLM_ATTEMPT_TIMEOUT"], \
        "総予算と 1 試行の上限がずれている"
    assert int(d["VLLM_STARTUP_BUDGET"]) >= 6500, \
        "実測の遅い側（4,650 秒）に余裕が無い"


def test_for_kernel_matches_the_ledger():
    """`for_kernel` は台帳の値だけを使い、上書きは明示した分だけ効く。"""
    from kbuild import metadata

    kernel = spec.find("step2-vllm-qwen38-27b-cloudflared-script")
    meta = metadata.for_kernel(kernel)
    assert meta["id"] == kernel.id
    assert meta["title"] == kernel.title
    assert meta["kernel_type"] == kernel.kind
    assert meta["enable_gpu"] is kernel.gpu
    assert tuple(meta["dataset_sources"]) == kernel.datasets
    metadata.validate(meta)

    # script 版は `code_file` を明示で上書きできる（台帳は main.py だが、
    # notebook 側の台帳エントリを流用する場合に要る）。
    assert metadata.for_kernel(kernel, code_file="x.py")["code_file"] == "x.py"
    # gpu=False は「上書きしない」と区別される（None だけが既定に落ちる）
    assert metadata.for_kernel(kernel, gpu=False)["enable_gpu"] is False
