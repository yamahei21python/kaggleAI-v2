"""LMDeploy / SGLang の起動引数と生成物を検査する（CPU のみ）。

GPU 実測はできないので、ここでは「引数の組み立て」「生成物の形」「決定性」を固定する。
"""
import ast
import hashlib
import pathlib
import sys

from kbuild import serve_script, spec
from runtime.engines import lmdeploy, sglang

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))
import build_engine_serve as bes  # noqa: E402


def test_lmdeploy_argv():
    argv = lmdeploy.kc_lmdeploy_argv("python", "/m", tp=2, port=8000)
    assert argv[:6] == ["python", "-m", "lmdeploy", "serve", "api_server", "/m"]
    assert "--backend" in argv and argv[argv.index("--backend") + 1] == "pytorch"
    assert argv[argv.index("--tp") + 1] == "2"
    assert argv[argv.index("--server-port") + 1] == "8000"
    assert argv[argv.index("--session-len") + 1] == "8192"


def test_sglang_argv_defaults_for_t4():
    """公式の sm75+ 回避策: attention=triton / sampling=pytorch。"""
    argv = sglang.kc_sglang_argv("python", "/m", tp=2, port=8000)
    assert argv[:3] == ["python", "-m", "sglang.launch_server"]
    assert argv[argv.index("--attention-backend") + 1] == "triton"
    assert argv[argv.index("--sampling-backend") + 1] == "pytorch"
    assert "--disable-cuda-graph" in argv
    assert argv[argv.index("--tp") + 1] == "2"


def test_sglang_sampling_backend_can_be_disabled():
    argv = sglang.kc_sglang_argv("python", "/m", sampling_backend=None)
    assert "--sampling-backend" not in argv


def test_sglang_graph_can_be_enabled():
    argv = sglang.kc_sglang_argv("python", "/m", disable_cuda_graph=False)
    assert "--disable-cuda-graph" not in argv


def test_pip_pins_version():
    assert lmdeploy.kc_lmdeploy_pip_argv("python")[-1] == "lmdeploy==0.18.0"
    # SGLang は**複数ステップ**。最後は確認なので、どこかのステップに入っていればよい。
    steps = sglang.kc_sglang_pip_steps("python")
    assert any("sglang==0.5.22" in " ".join(s) for s in steps)


def test_sglang_install_installs_rust_first():
    """既定の筋は「Rust を入れてから素の pip」。

    実測 2026-11-01（`sglang-install-probe`・GPU 0 秒）:
    rustup 9.4s + pip 389.8s で `import sglang` と `--help` が通った。
    Rust を入れないと `outlines_core==0.1.26`（cp313 wheel 無し）の
    sdist ビルドで落ちる。
    """
    steps = sglang.kc_sglang_pip_steps("python")
    assert len(steps) == 3
    assert steps[0][0] == "bash" and "rustup.rs" in steps[0][2]
    # pip 側も `bash -lc` で包む（rustup は PATH を書き換えない）
    assert steps[1][0] == "bash"
    assert ".cargo/env" in steps[1][2]


def test_sglang_install_verifies_torchaudio_last():
    """最後のステップは「確認と修理」。install が通っても起動で落ちる。

    実測 2026-11-01（v3）: install 388s で成功 → 起動直後に
    `PyTorch has CUDA version 13.0 whereas TorchAudio has CUDA version 12.8`。
    ステップの出力を捨てていたので、どの版が入ったか追えなかった。
    """
    steps = sglang.kc_sglang_pip_steps("python")
    fix = steps[-1]
    assert fix[1] == "-c", "最後のステップが python -c になっていない"
    src = fix[2]
    compile(src, "<torchaudio-fix>", "exec")
    assert "torchaudio" in src and "torch.version.cuda" in src
    assert "torchaudio==2.11.0" in src
    assert "--force-reinstall" in src
    # 修理したあとに**もう一度**確かめる（黙って通さない）
    assert src.count("_check()") >= 2
    assert "sys.exit(0 if ok else 1)" in src


def test_sglang_install_pin_route_orders_outlines_core_first():
    """筋 B（Rust 無し）は順序が本質。

    `outlines` を先に入れると `outlines_core==0.1.26` を取りに行って
    同じ所で死ぬ。`outlines-core==0.1.17` は cp313 wheel がある唯一の 0.1.x。
    """
    steps = sglang.kc_sglang_pip_steps("python", route="pin")
    flat = [" ".join(s) for s in steps]
    core = next(i for i, c in enumerate(flat) if "outlines-core==" in c)
    outlines = next(i for i, c in enumerate(flat) if "outlines==0.1.11" in c)
    assert core < outlines, f"順序が逆: {flat}"
    assert "0.1.17" in flat[core]


def test_sglang_install_route_can_be_switched_by_env(monkeypatch):
    """`KC_SGLANG_ROUTE` で実行時に切り替えられる（生成コードに式で埋まる）。"""
    monkeypatch.setenv("KC_SGLANG_ROUTE", "pin")
    assert len(sglang.kc_sglang_pip_steps("python")) == 5
    monkeypatch.setenv("KC_SGLANG_ROUTE", "rust")
    assert len(sglang.kc_sglang_pip_steps("python")) == 3


def test_ready_urls():
    assert lmdeploy.kc_lmdeploy_ready_url(8000).endswith("/v1/models")
    assert "8000" in sglang.kc_sglang_ready_url(8000)


def test_spec_entries_exist():
    for name in ("step2-lmdeploy-qwen38-27b-cloudflared",
                 "step2-sglang-qwen38-27b-cloudflared"):
        kernel = spec.find(name)
        assert kernel.gpu is True
        assert kernel.builder == "build_engine_serve"
        assert kernel.metadata_path.is_file()


def _render(tmp, name, smoke=False):
    engine = bes.ENGINES[name]
    serve_script.write(
        pathlib.Path(tmp) / name, engine, bes.METADATA[name],
        run_minutes=60,
        runtime="\n# runtime\n",
        proxy_b64="", html_b64="", gist_id="g", gist_owner="o",
        smoke=smoke,
    )
    return (pathlib.Path(tmp) / name / "main.py").read_text()


def test_generated_shape_and_determinism(tmp_path):
    for name, call in (("lmdeploy", "kc_lmdeploy_argv"),
                       ("sglang", "kc_sglang_argv")):
        body = _render(tmp_path / "a", name)
        again = _render(tmp_path / "b", name)
        assert hashlib.sha256(body.encode()).hexdigest() == \
            hashlib.sha256(again.encode()).hexdigest()
        ast.parse(body)
        assert f'ENGINE = "{name}"' in body
        assert f"{call}(" in body
        assert '"spec_decode": False' in body
        assert "kc_start_proxy(" in body
        assert "kc_keepalive_expired(" in body
        assert "names, mems = kc_detect_gpu(" in body   # tuple を dict 扱いしない
        assert 'data["api_pass"]' not in body
        assert '"API_PASS": pass_' in body          # proxy へは env で渡す
        assert "releases/latest" not in body
        # Secrets は env 優先 + kaggle_secrets フォールバック
        assert "UserSecretsClient" in body
        # 異常終了でもプロセスを残さない
        assert "def stop_all(" in body
        assert "kc_stop_processes(" in body
        assert "finally:" in body
        # 公開 /stop がエンジンも落とす
        assert '"--stop-pid"' in body


def test_smoke_variant_skips_tunnel(tmp_path):
    for name in ("lmdeploy", "sglang"):
        body = _render(tmp_path / "smoke" / name, name, smoke=True)
        assert "SMOKE = True" in body
        # smoke は proxy / tunnel より前に return する
        head = body.split("def run(procs):")[1]
        assert head.index("if SMOKE:") < head.index("start_tunnel()")
        assert '"smoke"' in head
        ast.parse(body)


def test_normal_variant_serves(tmp_path):
    body = _render(tmp_path / "normal", "sglang")
    assert "SMOKE = False" in body


def _render_tunnel(tmp, tunnel, name="sglang"):
    engine = bes.ENGINES[name]
    serve_script.write(
        pathlib.Path(tmp) / tunnel, engine, bes.METADATA[name],
        run_minutes=60,
        runtime="\n# runtime\n",
        proxy_b64="", html_b64="", gist_id="g", gist_owner="o",
        tunnel=tunnel,
    )
    return (pathlib.Path(tmp) / tunnel / "main.py").read_text()


def test_tunnel_defaults_to_cloudflared(tmp_path):
    """既定は cloudflared（エンジン辞書に tunnel を書かなくてよい）。"""
    body = _render_tunnel(tmp_path, "cloudflared")
    assert 'TUNNEL = "cloudflared"' in body
    assert "def start_cloudflared(" in body
    assert 'data["tunnel"] = TUNNEL' in body    # 設定と実装がずれても実装が正
    ast.parse(body)


def test_tunnel_has_no_ngrok_left(tmp_path):
    """ngrok の**実装**が骨格に残っていないこと（2026-11-01 に廃止）。

    残っていると「NGROK_TOKEN を Secrets に入れれば固定 URL になる」と
    読めてしまう。専用アカウントでは Secret が読めず、ビーコンで URL を
    配っても手前で起動できない（`notes/url-publishing.md`）。

    ただし**説明文には残す**。廃止の理由を書かないと、次に読む人が同じ
    判断をやり直すことになる。なので「文字として出てこない」ではなく
    「コードとして使われていない」を見る: AST から識別子と、docstring
    以外の文字列定数を拾い、そこに ngrok が無いことを確かめる。
    コメントは AST に載らないので、この判定に引っかからない。
    """
    body = _render_tunnel(tmp_path, "cloudflared")
    tree = ast.parse(body)

    docs = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        head = getattr(node, "body", [None])[0]
        if (isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant)
                and isinstance(head.value.value, str)):
            docs.add(id(head.value))

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            used.add(node.name)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            used.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            used.update(a.name for a in node.names)
        elif isinstance(node, ast.Constant) and id(node) not in docs:
            used.add(str(node.value))

    hits = sorted(u for u in used if "ngrok" in u.lower())
    assert not hits, f"ngrok が実装に残っている: {hits}"


def test_tunnel_rejects_ngrok(tmp_path):
    """`tunnel="ngrok"` は生成時に落とす（cloudflared で静かに動かさない）。"""
    import pytest
    with pytest.raises(ValueError, match="未対応"):
        _render_tunnel(tmp_path, "ngrok")


def test_tunnel_rejects_unknown(tmp_path):
    """綴り間違いは生成時に落とす（Kaggle で初めて気づくのを避ける）。"""
    import pytest
    with pytest.raises(ValueError, match="未対応"):
        _render_tunnel(tmp_path, "tailscale")


def test_from_cells_preserves_order_and_comments():
    """セル連結は順序を保ち、markdown は節コメントに落ちる。"""
    from kbuild import pack_script
    cells = [
        ("markdown", "# 手順\n\n説明文"),
        ("code", "a = 1\nprint(a)"),
        ("code", "b = a + 1"),
    ]
    body = pack_script.from_cells(cells, header="#!/usr/bin/env python3\n")
    assert body.index("a = 1") < body.index("b = a + 1")
    assert "# ── 手順" in body
    assert "説明文" not in body          # markdown 本文は落とす
    compile(body, "<main>", "exec")


def test_from_cells_accepts_notebook_dict_cells():
    """pack_nb.write_cells 系（dict + source が行リスト）も受ける。"""
    from kbuild import pack_script
    cells = [{"cell_type": "code", "source": ["x = 1\n", "y = 2\n"]}]
    body = pack_script.from_cells(cells)
    assert "x = 1" in body and "y = 2" in body


def test_from_cells_rejects_ipython_magic():
    """マジックは黙って壊れた main.py にせず、その場で落とす。"""
    import pytest
    from kbuild import pack_script
    with pytest.raises(ValueError, match="マジック"):
        pack_script.from_cells([("code", "!pip install foo")])
