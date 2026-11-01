"""テンプレート（`kbuild/serve_script.py`）の防御を固定する回帰テスト。

どれも「気付くまでに 1 往復 6〜46 分かかった」種類のもの。GPU を使わずに
ここで固定しておく。`TEMPLATE` は `$PLACEHOLDER` を含む `string.Template` なので
ファイル全体は exec できない。関数の切り出しは `tests/_cells.py` に集約した
（以前はこのファイルが独自実装を持っていた）。

vLLM の**セル**（install / serve）の防御は `tests/test_vllm_cells.py` にある。
ここは「全エンジン共通の足場」だけを持つ。
"""
import glob
import os
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import _cells  # noqa: E402
from kbuild import serve_script  # noqa: E402
from runtime.engines import lmdeploy, sglang  # noqa: E402

ROOT = _cells.ROOT
SRC = serve_script.TEMPLATE.template


def _load_func(name, **names):
    """テンプレートから関数を 1 つ取り出す（`$` 置換前でも exec できる形）。

    `names` で `glob` を差し替えられるよう、既定値は dict を更新する形で入れる
    （キーワードで二重に渡すと TypeError になる）。
    """
    ns = {"os": os, "glob": glob}
    ns.update(names)
    return _cells.exec_def(SRC, name, **ns)


# ── wait_http: エンジンが死んだら待ち続けない ──────────────────
class _Clock:
    """実時間を進めない時計。timeout=2400 のテストを一瞬で終わらせる。"""

    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s
        self.slept += s


def _make_wait_http(*, engine_alive, ready=False, clock=None):
    clock = clock or _Clock()

    def http_json(url, auth=None, timeout=10):
        if ready:
            return 200, {}
        raise OSError("[Errno 111] Connection refused")

    def kc_pids(pattern, timeout=5):
        return [4242] if engine_alive else []

    return _load_func(
        "wait_http",
        http_json=http_json,
        kc_pids=kc_pids,
        ENGINE_PATTERN="-f lmdeploy",
        time=clock,
    ), clock


def test_wait_http_gives_up_when_engine_dies():
    """engine が消えたら timeout(2400s) を待たずに諦める。

    実測: lmdeploy は 6 分で CUDA OOM 死したのに待機が 40 分続き、1 往復 46 分。
    """
    fn, clock = _make_wait_http(engine_alive=False)
    with pytest.raises(RuntimeError) as e:
        fn("http://127.0.0.1:8000/v1/models", timeout=2400)
    assert "エンジンが終了した" in str(e.value)
    assert clock.slept < 200, f"待ちすぎ（{clock.slept}s 経過）"


def test_wait_http_keeps_waiting_while_engine_alive():
    """プロセスが生きている間は従来どおり待つ（誤検知しない）。"""
    fn, clock = _make_wait_http(engine_alive=True)
    with pytest.raises(RuntimeError) as e:
        fn("http://127.0.0.1:8000/v1/models", timeout=2400)
    assert "待機タイムアウト" in str(e.value)
    assert clock.slept >= 2400


def test_wait_http_returns_elapsed_when_ready():
    fn, _ = _make_wait_http(engine_alive=True, ready=True)
    assert fn("http://127.0.0.1:8000/v1/models", timeout=2400) == 0.0


# ── find_model: 量子化の持ち方 2 通りを受ける ──────────────────
class _FakeGlob:
    """`/kaggle/input/...` を tmp ディレクトリに読み替える glob。"""

    def __init__(self, root):
        self.root = str(root)

    def glob(self, pattern):
        # `/kaggle/input/` までを root に読み替える（`datasets/*/*` の階層を保つ）
        if pattern.startswith("/kaggle/input/"):
            pattern = os.path.join(self.root, pattern[len("/kaggle/input/"):])
        return sorted(glob.glob(pattern))


def _find_model(root):
    return _load_func("find_model", glob=_FakeGlob(root))()


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def test_find_model_accepts_embedded_quantization(tmp_path):
    """`config.json` に quantization_config を埋め込む AWQ を見つけられる。

    実測: この Dataset には `quantization_config.json` が無いため、lmdeploy が
    `モデル Dataset が見つからない` で 30 秒で落ちた（Dataset はマウント済み）。
    """
    d = tmp_path / "datasets" / "yamahei21python" / "awq-mtp"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert _find_model(tmp_path) == str(d)


def test_find_model_still_accepts_split_quantization_file(tmp_path):
    """従来の `quantization_config.json` 形式も壊していない。"""
    d = tmp_path / "datasets" / "owner" / "split"
    _touch(d / "config.json")
    _touch(d / "quantization_config.json")
    assert _find_model(tmp_path) == str(d)


def test_find_model_rejects_config_without_weights(tmp_path):
    """config.json だけのディレクトリ（= 重みが無い）は拾わない。"""
    d = tmp_path / "datasets" / "owner" / "broken"
    _touch(d / "config.json")
    assert _find_model(tmp_path) is None


# ── stage_env: 親プロセスで torch を import しない ────────────
def test_stage_env_does_not_import_torch_in_parent():
    """親が先に torch を握ると、あとで差し替えたときに壊れる。

    実測: メモリ上 2.11 / ディスク上 2.13 が混ざり
    `TypeError: Config() got an unexpected keyword argument 'deprecated'`。
    子プロセスでは成功するのに親だけ落ちる、という形で 8 往復溶かした。
    """
    body = _cells.slice_def(SRC, "stage_env")
    assert "import torch" not in body, "親プロセスで torch を import している"
    assert "import json, torch" in body, "子プロセスへの委譲が無い"


# ── ENGINE_PATTERN: 起動コマンドと一致していないと誤って諦める ──
# `wait_http` は「ENGINE_PATTERN に一致するプロセスが 3 回連続で無い」をエンジン死亡と
# 判断する。パターンが起動コマンドと食い違うと、**正常に起動している最中に 75 秒で
# 諦める**回帰になる。engine を足すときはこの表にも足すこと。
_LAUNCH_SOURCES = {
    "vllm": (
        "-f vllm",
        lambda: (ROOT / "kaggle/step2-vllm/cells/03_vllm_serve.py").read_text(),
    ),
    "lmdeploy": (
        "-f lmdeploy",
        lambda: " ".join(lmdeploy.kc_lmdeploy_argv("python", "/m")),
    ),
    "sglang": (
        "-f sglang.launch_server",
        lambda: " ".join(sglang.kc_sglang_argv("python", "/m")),
    ),
    "llama-server": (
        "-f llama-server",
        lambda: (ROOT / "kaggle/llama-server/cells/03_llama_serve.py").read_text(),
    ),
    "ollama": (
        "-x ollama",
        lambda: (ROOT / "kaggle/step2/cells/03_ollama_serve.py").read_text(),
    ),
    "exl3": (
        "-f chat_server.py",
        lambda: (ROOT / "kaggle/build_exl3_serve.py").read_text(),
    ),
}


@pytest.mark.parametrize("engine", sorted(_LAUNCH_SOURCES))
def test_engine_pattern_matches_its_launch_command(engine):
    pattern, source = _LAUNCH_SOURCES[engine]
    needle = pattern.split()[-1]          # "-f sglang.launch_server" → "sglang.launch_server"
    assert needle in source(), (
        f"{engine}: ENGINE_PATTERN={pattern!r} が起動コマンドに含まれない。"
        " wait_http が起動中に『エンジンが終了した』と誤判定する"
    )



# ── 複数ステップ install: SGLang は 1 回の pip では通らない ──────────
# 実測 2026-11-01: `step2-sglang-…` は 297 秒で `pip install 失敗 (1)`。
# `install_tail` は `Failed building wheel for outlines_core`。原因は
# `sglang==0.5.22 → outlines==0.1.11 → outlines_core==0.1.26` の完全固定で、
# 0.1.26 に cp313 の wheel が無いこと。**1 回の pip では回避できない**ので、
# ステップを分けられる足場が要る。

class _Proc:
    def __init__(self, rc):
        self.returncode = rc
        self.stdout = "out"
        self.stderr = "boom" if rc else ""


def _stage_install(steps, rcs):
    """生成コードの `stage_install()` を、`sh` を差し替えて実行する。"""
    import time as _time

    calls, saves = [], []

    def sh(argv, timeout=None):
        calls.append(list(argv))
        rc = rcs[len(calls) - 1] if len(calls) <= len(rcs) else 0
        return _Proc(rc), 1.0

    data = {}
    fn = _cells.exec_def(
        serve_script.EXTERNAL_ENGINE_CODE.template, "stage_install",
        PIP_STEPS=steps, sh=sh, tail=lambda s, n=30: s,
        save=lambda: saves.append(dict(data)), data=data, time=_time)
    return fn, data, calls, saves


def test_install_template_takes_a_list_of_steps():
    """`PIP_ARGV`（1 回）に戻っていないこと。

    1 回に戻すと SGLang の回避（outlines_core を先に入れる）が書けなくなる。
    """
    ext = serve_script.EXTERNAL_ENGINE_CODE.template
    assert "PIP_STEPS" in ext
    assert "PIP_ARGV" not in ext


def test_install_runs_every_step_in_order():
    steps = [["pip", "a"], ["pip", "b"], ["pip", "c"]]
    fn, data, calls, _ = _stage_install(steps, [0, 0, 0])
    fn()
    assert calls == steps, f"順序か回数が違う: {calls}"
    assert [s["index"] for s in data["install_steps"]] == [1, 2, 3]
    assert data["install_seconds"] == 3.0


def test_install_stops_at_the_failing_step_and_says_which():
    """落ちたステップ番号を例外に入れる。

    これが無いと、また「どの pip で落ちたか分からない」状態に戻る
    （実際 297 秒のログに残っていたのは `pip install 失敗 (1)` だけだった）。
    """
    steps = [["pip", "ok"], ["pip", "bad"], ["pip", "never"]]
    fn, data, calls, _ = _stage_install(steps, [0, 1])
    with pytest.raises(RuntimeError) as ei:
        fn()
    msg = str(ei.value)
    assert "step 2/3" in msg, msg
    assert "pip bad" in msg, msg
    assert len(calls) == 2, f"失敗後も続けて実行している: {calls}"
    assert "boom" in data["install_tail"]
    assert data["install_steps"][-1]["rc"] == 1


def test_install_keeps_going_after_a_successful_step():
    """途中で `save()` していること（失敗したときに前段の所要が残る）。"""
    steps = [["pip", "a"], ["pip", "b"]]
    fn, data, calls, saves = _stage_install(steps, [0, 1])
    with pytest.raises(RuntimeError):
        fn()
    assert len(saves) >= 2, "ステップごとに save していない"
    assert saves[0]["install_steps"][0]["rc"] == 0
