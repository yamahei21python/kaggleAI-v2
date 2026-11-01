"""notebook 配信カーネルの「1 本化」が保たれているか検査する。

3 系統（Ollama / llama-server / vLLM）は `kaggle/serve-cells/` の同じセルを
共有する。ここが崩れると、新しいエンジンを足したときに配管だけ古い実装が
残る — しかもセル単位では正しい Python なので CI は通ってしまう
（実際に exl3 の script テンプレートで同じことが起きていた）。

守る不変条件:

  1. 共有セルは 3 系統に **バイト一致で** 埋め込まれている
  2. 埋め込み順序が全系統で同じ（proxy → tunnel → gist → verify → 接続情報 → keepalive）
  3. 共有セルはエンジン名を持たない（エンジン差は settings の能力表だけ）
  4. 能力表（ENGINE_PROCS / CLEANUP_TARGETS / engine_cleanup）の定義は 1 か所だけ
"""
import json
import pathlib

from kbuild.serve_notebook import CELLS_DIR

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (ラベル, notebook, トンネル)
# ngrok 系の 3 本は 2026-11-01 に RETIRED にした（`notes/url-publishing.md`）。
# notebook は履歴としてディスクに残すが、共有セル `tunnel_ngrok.py` を消した
# のでもう再生成できない。監査対象は「今もビルドできる配信カーネル」だけ。
NOTEBOOKS = [
    ("ollama/cloudflared",
     "kaggle/step2/step2-orcarouter-qwen38-27b-cloudflared.ipynb", "cloudflared"),
    ("vllm/cloudflared",
     "kaggle/step2-vllm-kernel/cloudflared/"
     "step2-vllm-qwen38-27b-cloudflared.ipynb", "cloudflared"),
]

# 配管セルの並び。ここが「カーネルの形」そのもの。
ORDER = ["proxy.py", "{tunnel}", "gist.py", "verify.py",
         "connection.py", "keepalive.py"]

# エンジン名。共有セルに現れたら「1 本化が崩れた」合図。
ENGINE_WORDS = ("ollama", "vllm", "llama-server", "llama_server", "lmdeploy",
                "sglang", "exl3")


def _code(path):
    nb = json.load(open(ROOT / path))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _shared(name):
    return (CELLS_DIR / name).read_text(encoding="utf-8")


def test_shared_cells_are_embedded_verbatim():
    print("\n=== 共有セルが 3 系統にバイト一致で入っている ===")
    for label, path, tunnel in NOTEBOOKS:
        cells = _code(path)
        names = [n.format(tunnel=f"tunnel_{tunnel}.py") for n in ORDER]
        for name in names:
            src = _shared(name)
            assert src in cells, f"{label}: 共有セル {name} が埋め込まれていない"
        print(f"  {label}: {len(names)} セル OK")


def test_shared_cell_order_is_identical():
    print("\n=== 配管セルの順序が全系統で同じ ===")
    for label, path, tunnel in NOTEBOOKS:
        cells = _code(path)
        names = [n.format(tunnel=f"tunnel_{tunnel}.py") for n in ORDER]
        pos = [cells.index(_shared(n)) for n in names]
        assert pos == sorted(pos), f"{label}: 配管セルの順序が違う {pos}"
        print(f"  {label}: {pos}")


def test_shared_cells_have_no_engine_names():
    """エンジン差は settings の能力表だけ。共有セルに if エンジン名 を書かない。"""
    print("\n=== 共有セルにエンジン名が無い ===")
    for name in sorted(p.name for p in CELLS_DIR.glob("*.py")):
        src = _shared(name).lower()
        hit = [w for w in ENGINE_WORDS if w in src]
        assert not hit, (f"{name} にエンジン名 {hit} が入っている。"
                         "エンジン差は settings の能力表（ENGINE_* / engine_cleanup）"
                         "で表現する")
        print(f"  {name}: なし")


def test_capability_table_is_defined_once():
    """ENGINE_PROCS / CLEANUP_TARGETS / engine_cleanup は settings だけが持つ。"""
    print("\n=== 能力表の定義は 1 か所だけ ===")
    markers = ("ENGINE_PROCS =", "CLEANUP_TARGETS =", "def engine_cleanup(")
    for label, path, _tunnel in NOTEBOOKS:
        cells = _code(path)
        for marker in markers:
            n = sum(1 for s in cells if marker in s)
            assert n == 1, f"{label}: {marker!r} の定義が {n} 個ある（1 個であるべき）"
        print(f"  {label}: 各 1 個 OK")


def test_settings_defines_the_contract():
    """共有セルが読む名前を settings セルが定義している（名前の取りこぼし検出）。"""
    print("\n=== settings が契約名を定義している ===")
    required = [
        "PORT", "UPSTREAM_PORT", "PROXY_PORT", "CHAT_PATH", "PROXY_PATH",
        "PROXY_AUTH", "PROXY_EXTRA_ARGS", "PROXY_CHECK_PATH",
        "MODEL_NAME", "MODEL_DISPLAY", "ENGINE_DISPLAY", "CONTEXT_LENGTH",
        "GIST_ID_DEFAULT", "GIST_OWNER",
        "MAX_MINUTES", "KEEPALIVE_RESET_MINUTES", "TUNNEL_CHECK_EVERY",
        "ENGINE_LABEL", "ENGINE_PROCS", "CLEANUP_TARGETS", "ENGINE_LOG",
        "OUT_DIRS", "save_output", "fail",
    ]
    for label, path, _tunnel in NOTEBOOKS:
        cells = _code(path)
        # settings は ENGINE_PROCS を持つセル（他のセルは読むだけ）
        settings = next(s for s in cells if "ENGINE_PROCS =" in s)
        missing = [n for n in required if n not in settings]
        assert not missing, f"{label}: settings に {missing} が無い"
        print(f"  {label}: {len(required)} 名 OK")
