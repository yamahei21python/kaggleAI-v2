"""CPU プローブ（`kaggle/cpu-tests/` + `kaggle/whl-probe/`）の契約を固定する。

これらは **GPU クォータ 0 秒**で回す代わりに「GPU でしか分からないこと」を
調べない代物なので、実行結果そのものは Kaggle でしか取れない。
ここで固定するのは**実行しなくても壊れたと分かる契約**:

- `enable_gpu: false`（これが true になると、無料のつもりがクォータを食う）
- 台帳（`kbuild/spec.py`）と metadata の一致
- ケース表が README の約束を満たしているか（対照実験の有無など）
"""
import ast
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kbuild import spec  # noqa: E402

# 台帳の id → 出力ディレクトリ
PROBES = {
    "tiny-llama-probe": ROOT / "kaggle/cpu-tests/tiny-llama",
    "tiny-ollama-probe": ROOT / "kaggle/cpu-tests/tiny-ollama",
    "vllm-flag-probe": ROOT / "kaggle/cpu-tests/vllm-flags",
    "vllm-whl-probe": ROOT / "kaggle/whl-probe",
    "sglang-install-probe": ROOT / "kaggle/cpu-tests/sglang-probe",
}

# モデルを実際に走らせるプローブ（gpu_unknown を出す約束がある）
MODEL_PROBES = ("tiny-llama-probe", "tiny-ollama-probe")


def _metadata(name):
    return json.loads((PROBES[name] / "kernel-metadata.json").read_text())


def _cases(path):
    """`CASES = [...]` を AST から取り出す（実行はしない）。"""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CASES":
                    return ast.literal_eval(node.value)
    return None


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probe_metadata_disables_gpu(name):
    """CPU プローブは GPU を要求しない（要求した瞬間に無料ではなくなる）。"""
    meta = _metadata(name)
    assert meta["enable_gpu"] is False, f"{name} が GPU を要求している"
    assert meta["kernel_type"] == "script"
    assert meta["id"].endswith("/" + name)


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probe_is_cpu_in_ledger(name):
    """台帳側も gpu=False で、出力先が実ディレクトリと一致している。"""
    ledger = {k.name: k for k in spec.all_kernels()}
    assert name in ledger, f"{name} が台帳に無い"
    kernel = ledger[name]
    assert kernel.gpu is False, f"台帳の {name} が gpu=True"
    assert kernel.dir == PROBES[name], f"台帳の出力先が違う: {kernel.dir}"


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probe_writes_result_json(name):
    """結果は `/kaggle/working/result.json` に残す（`kaggle kernels output` で拾う）。"""
    src = (PROBES[name] / "main.py").read_text()
    assert "/kaggle/working" in src
    assert '"result.json"' in src


@pytest.mark.parametrize("name", MODEL_PROBES)
def test_model_probes_declare_gpu_unknown(name):
    """CPU で確定できない項目を明示する（README の設計メモ）。

    これが無いと、CPU の結果を GPU の結果と読み違える。
    """
    src = (PROBES[name] / "main.py").read_text()
    assert 'DATA["gpu_unknown"]' in src


def test_tiny_llama_cases_cover_documented_flags():
    """README が約束している 4 観点がケース表に入っているか。"""
    cases = _cases(PROBES["tiny-llama-probe"] / "main.py")
    assert cases, "CASES が取れない"
    names = [c[0] for c in cases]
    assert "base" in names, "基準ケースが無い（比較の土台）"

    flat = {c[0]: c[1] for c in cases}
    # KV 量子化は -fa とセットで検証する（片方だけでは意味が薄い）
    assert "--cache-type-k" in flat["kv_q8"]
    assert "--cache-type-v" in flat["kv_q8"]
    assert "-fa" in flat["kv_q8"]
    # MTP の spec-type
    assert "--spec-type" in flat["spec_mtp"]
    # 並列
    assert any(c[3] == 2 for c in cases), "-np 2 相当の並列ケースが無い"


def test_vllm_flag_cases_include_negative_control():
    """対照実験（存在しないフラグ）が入っていること。

    これが「受理」と判定されたら判定器そのものが壊れている。ケース表から
    消すと、以後すべての PASS が信用できなくなる。
    """
    cases = _cases(PROBES["vllm-flag-probe"] / "main.py")
    assert cases, "CASES が取れない"
    bogus = [c for c in cases if any(f.startswith("--definitely-not-a-flag")
                                     for f in c[1])]
    assert bogus, "対照実験（存在しないフラグ）がケース表から消えている"
    assert len(bogus) == 1, "対照実験は 1 本だけにする（増やすと判定がぼやける）"


def test_vllm_flag_cases_are_pairs():
    """各ケースは (名前, argv) の 2 要素。argv は空でない文字列のリスト。"""
    cases = _cases(PROBES["vllm-flag-probe"] / "main.py")
    for case in cases:
        assert len(case) == 2, f"形が違う: {case}"
        name, argv = case
        assert isinstance(name, str) and name
        assert isinstance(argv, list) and argv
        assert all(isinstance(a, str) for a in argv)


def test_vllm_flag_verdict_excludes_designed_negatives():
    """「弾かれて正解」のケースを ok の判定に混ぜない。

    実測 2026-10-31（v2）: 完走しているのに `ok=False` になった。原因は
    `--kv-cache-dtype q8_0`（存在しない値）を positive 側に数えていたこと。
    負のケースは `detector_works` 側で見る、という分担をここで固定する。
    """
    src = (PROBES["vllm-flag-probe"] / "main.py").read_text()
    assert "_NEGATIVE" in src, "負のケースの集合が無い（ok の意味が曖昧になる）"
    neg = next(ln for ln in src.splitlines() if ln.strip().startswith("_NEGATIVE"))
    assert "bogus" in neg and "kv_q8" in neg, (
        f"負のケースに bogus と kv_q8 の両方が要る: {neg.strip()}")
    # ok は「positive が全部 argparse を通る」かつ「負のケースが弾かれる」
    assert "detector_works" in src
    assert "positive" in src


# ---- SGLang install probe ----
# 実測 2026-11-01: `step2-sglang-…` は 297 秒で `pip install 失敗 (1)`。
# `install_tail` は `Failed building wheel for outlines_core`。**エンジンは未起動**。
# つまり「T4 で動かない」ではなく「まだ何も試していない」。ここで固定するのは、
# その切り分けを GPU でやらないための契約。

def _sglang_src():
    return (PROBES["sglang-install-probe"] / "main.py").read_text()


def test_sglang_probe_asks_pip_instead_of_asserting_no_wheel():
    """「cp313 の wheel が無い」を**pip に判定させる**。

    手で書いた事実は必ず古くなる（上流が wheel を足した瞬間に嘘になる）。
    `--only-binary=:all:` は「wheel が無ければ失敗」なので、推測が要らない。
    """
    src = _sglang_src()
    assert "--only-binary=:all:" in src, "wheel の有無を推測で書いている"
    assert "outlines-core" in src and "0.1.26" in src, "ブロッカーの版が消えている"


def test_sglang_probe_route_a_keeps_the_pin_intact():
    """筋 A は `--no-deps` を付けない（固定を壊さないのが取り柄）。

    ここに `--no-deps` が紛れると、筋 A と筋 B の違いが消えて
    「どちらが通ったか」が意味を失う。
    """
    src = _sglang_src()
    body = src.split("def route_a():")[1].split("def route_b():")[0]
    assert "--no-deps" not in body, "筋 A に --no-deps が入っている"
    assert "sglang=={SGLANG_VERSION}" in body


def test_sglang_probe_route_b_installs_outlines_core_first():
    """筋 B は順序が本質。`outlines` を先に入れると 0.1.26 を取りに行って同じ所で死ぬ。"""
    src = _sglang_src()
    body = src.split("def route_b():")[1].split("def main():")[0]
    core = body.index("outlines-core==")
    outlines = body.index('"outlines==0.1.11"')
    assert core < outlines, "outlines-core より先に outlines を入れている（順序が逆）"
    # 本体も依存も --no-deps で入れないと、解決が走って 0.1.26 に戻る
    assert body.count("--no-deps") >= 2


def test_sglang_probe_does_not_claim_an_sm75_verdict():
    """install が通っても sm75 は未検証。**そこを明示しないと読み違える**。"""
    src = _sglang_src()
    assert 'DATA["verdict"]' in src
    assert '"sm75_untested": True' in src, (
        "sm75 未検証であることが result.json に残らない")

