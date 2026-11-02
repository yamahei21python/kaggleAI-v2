#!/usr/bin/env python3
"""Step2 notebook builder (vLLM / AWQ+MTP): Qwen3.8-27B-Uncensored on Kaggle T4 x2

セルの並び・配管（proxy / tunnel / ビーコン / verify / 接続情報 / keepalive）は
`kbuild/serve_notebook.py` が唯一の定義。ここが持つのは **vLLM 固有の 4 つ**だけ:

    settings / install / launch / verify_checks  (+ warmup)

実体は `kaggle/step2-vllm/cells/*.py` に置く（ruff で構文チェックできるように）。

    python3 kaggle/build_step2_vllm.py                  # cloudflared
    python3 kaggle/build_step2_vllm.py --smoke          # 起動＋会話だけで終了

`STEP2_TUNNEL` は読まない（cloudflared 固定）。

出力: kaggle/step2-vllm-kernel/<tunnel>/step2-vllm-qwen38-27b-<tunnel>.ipynb
      smoke 版は kaggle/step2-vllm-kernel/smoke-<tunnel>/ へ別名で書く
      （push はしない。Dataset 作成完了後に手動で push する）
"""
import os
import pathlib
import sys

# ── 固定パラメータ ──────────────────────────────────────────
OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
DATASET_ID = f"{OWNER}/{DATASET_SLUG}"
HF_MODEL = "twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP"
SERVED_NAME = "qwen3.8-27b-awq"

TUNNEL = "cloudflared"

SLUG = f"step2-vllm-qwen38-27b-{TUNNEL}"

HERE = pathlib.Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parent))
from cli import config  # noqa: E402
from kbuild import embed, metadata, pack_nb, spec  # noqa: E402
from kbuild.serve_notebook import Flow  # noqa: E402
from kbuild.settings_cell import env_preamble as render_env  # noqa: E402
from runtime import proxy_source  # noqa: E402

CFG = config.load()

# smoke 版は通常版と干渉しない別ディレクトリへ。
# （push --smoke が通常生成物を上書きしないため。実稼働時は通常版を使う）
# トンネルごとにも分ける（smoke の cloudflared 実行が通常版を潰さない）。
SMOKE_OUT_DIR = HERE / "step2-vllm-kernel" / f"smoke-{TUNNEL}"

# smoke: トンネル / ビーコン公開 / keepalive を省略し、起動と会話だけで終了する。
# CPU 側（dryrun / proxy / keepalive のテスト）で確認済みの経路を GPU で
# なぞらないためのフラグ。`cli push ... --smoke` が `--smoke` を渡す。
# check_notebooks は引数を渡せないので KC_SMOKE=1 でも切り替えられる。
SMOKE = "--smoke" in sys.argv or os.environ.get("KC_SMOKE") == "1"

# prepare-vllm-dataset.ipynb / kernel-metadata.json と同じディレクトリに置くと
# メタデータが衝突するので、kernel 用は専用ディレクトリへ分ける。
# トンネルごとにディレクトリを分ける。kernel-metadata.json は push 時の
# 必須ファイル名なので、同居させると片方の設定が上書きされる。
OUT_DIR = SMOKE_OUT_DIR if SMOKE else HERE / "step2-vllm-kernel" / TUNNEL
OUT = OUT_DIR / f"step2-vllm-qwen38-27b-{TUNNEL}.ipynb"

# 単一ソース（kaggle/step2/*）をそのまま埋め込む。notebook 側で二重管理しない。
PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(HERE.parent / "llama-chat.html")

CELLS_DIR = HERE / "step2-vllm" / "cells"


def _cell(name):
    return (CELLS_DIR / name).read_text(encoding="utf-8")


def _tpl(src, **kw):
    """__KEY__ 置換。f-string だと notebook 側の dict リテラルを
    すべて二重 brace にする必要があり事故るので、置換で渡す。"""
    for k, v in kw.items():
        src = src.replace(f"__{k}__", str(v))
    return src


# ── title ───────────────────────────────────────────────────
cf_bullet = ("   - `my-cloudflared-bin`（cloudflared 40 MB）"
             "— 無いと毎回 GitHub から落とす")
secret_note = (
    """> ✅ **Secrets の設定は不要です。**（cloudflared はアカウント不要）
>
> 🔓 認証はありません（URL を知れば誰でも使えます）。
> 他人に使われたくないときは環境変数 `API_PASS` を設定してください。
>
> ⚠ 公開URLは **実行のたびに変わります**（quick tunnel の仕様）。
> 現在の URL はビーコン（`python3 -m cli beacon`）から読めます。""")

TITLE = _tpl("""# 🚀 Step 2 (vLLM): Qwen3.8-27B AWQ + __TUNNEL__ 公開APIサーバー

`__HFMODEL__` を **vLLM** で T4 x2 に載せ、__TUNNEL__ で公開します。

---

### 前提条件（これだけ）

1. **Accelerator**: `GPU T4 x2`
2. **Internet**: `On`
3. **Add Data**: 以下を追加
   - `__DATASET__`（AWQ 4bit モデル 約19 GB）
__CFBULLET__
__SECRET__

---

### Ollama 版との違い

| 項目 | Ollama 版 | このノート（vLLM） |
|---|---|---|
| 量子化 | GGUF Q4_K_M | AWQ W4A16 |
| 推論 | llama.cpp | vLLM（PagedAttention） |
| MTP | なし | `--speculative-config` で有効化（失敗時は自動で無効） |
| 速度 | 13.7 t/s 実測 | T4 x2 では 10〜25 t/s 見込み |

> ℹ T4 は sm75 のため **Marlin カーネルと FP8 KV cache は使えません**。
> AWQ は Triton 実装にフォールバックします。
""", TUNNEL=TUNNEL, HFMODEL=HF_MODEL, DATASET=DATASET_ID,
    CFBULLET=cf_bullet, SECRET=secret_note)

# 既定ノブ（`VLLM_BAKE_ENV`）は設定セルの**前**に置く。後ろだと、設定セルや
# install セルが既定値で動き出してから書き換えることになり、効かない。
SETTINGS = render_env(spec.find(SLUG).bake_env) + _tpl(
    _cell("00_settings.py"),
    DATASET_SLUG=DATASET_SLUG, SERVED=SERVED_NAME, TUNNEL=TUNNEL,
    PROXY_B64=PROXY_SRC_B64, CHAT_B64=CHAT_SRC_B64,
    KAGGLE_OWNER=CFG["KAGGLE_OWNER"],
)

LAUNCH_MD = '''---
### 🔥 vLLM サーバー起動（初回は 5〜10 分）

起動条件は **上から順に試行**し、失敗したら次へ落ちます。

| # | MTP | max-model-len | 備考 |
|---|---|---|---|
| 1 | ON | 32768 | 本命 |
| 2 | ON | 16384 | KV 不足時 |
| 3 | OFF | 16384 | MTP 非対応アーキのとき |
| 4 | OFF | 8192 | 最小構成 |
| 5 | OFF | 8192 | V0 エンジン（T4 用フォールバック） |

`--speculative-config` などのフラグは `vllm serve --help` を見て
**存在するものだけ**付けます（バージョン差で即死しないため）。

> 🎛 プロキシが `temperature<=0` を 0.7 に、`repetition_penalty` を 1.05 に
> 矯正します（`--enforce-sampling`）。temperature=0 で崩壊させないため。
'''

flow = Flow(
    title=TITLE,
    settings=SETTINGS,
    install=_cell("01_install_vllm.py"),
    launch=_cell("03_vllm_serve.py"),
    verify_checks=_cell("06_checks.py"),
    warmup=_cell("04_warmup.py"),
    tunnel=TUNNEL,
    install_md="---\n### 📡 vLLM インストール + プロキシ配置\n\n"
               "vLLM の wheel 取得に 5〜10 分かかります（2回目以降は Kaggle の"
               "キャッシュで短縮）。\n",
    launch_md=LAUNCH_MD,
    warmup_md="---\n### 🔥 ウォームアップ（初回生成は CUDA graph 構築で遅い）\n",
    verify_md="---\n### ✅ 公開URL経由の疎通確認（vLLM 版）\n\n"
              "UI / models / streaming / OpenAI互換 / GPU常駐 / 会話 の6点。\n",
    keepalive_md="""---
### 🔋 キープアライブ

既定 **30 分**で自動終了します（`MAX_MINUTES`）。上限まで延長できます。

> 実行が完了しないと `/kaggle/working` の zip が生成されず、ログも成果物も
> 取得できません。検証は必ず完了させてから `kaggle kernels output` で行います。

```bash
kaggle kernels output __SLUG__ -p out/
cat out/verify_result.txt out/connection_info.txt
```
""".replace("__SLUG__", f"{OWNER}/step2-vllm-qwen38-27b-{TUNNEL}"),
    # `beacon` は URL 配布に使う。既定の束に無いので明示する。
    runtime=("net", "proc", "storage", "keepalive", "proxy", "beacon"),
)

if SMOKE:
    # ここから先（トンネル / ビーコン / keepalive / 後片付け）は CPU 側で検証済みなので
    # 生成しない。ウォームアップの code セルまでで打ち切る。
    cells = flow.cells()
    _warm = max(i for i, (_kind, src) in enumerate(cells) if src == flow.warmup)
    cells = cells[: _warm + 1] + [
        ("markdown", "---\n### 🏁 smoke: ここで終了\n\n"
                     "起動と会話（ウォームアップセルの疎通確認）まで確認済み。\n"
                     "トンネル / ビーコン公開 / keepalive は CPU 側で検証済みのため省略します。\n"),
        ("code", '''# smoke: プロセスを止めて終了（公開しない）
print()
print("=" * 56)
print("🏁 smoke: トンネル / ビーコン / keepalive は省略")
print("=" * 56)
left = kc_stop_processes([
    ("-f auth_proxy.py", "auth_proxy"),
    ("-f vllm serve", "vllm"),
    ("-f VLLM::EngineCore", "engine-core"),
])
# 残存を「表示だけ」で流すと、GPU を掴んだまま完走扱いになる。
# 実際に解放されたかまで見て、駄目ならカーネルを失敗させる。
_idle = kc_wait_gpu_idle(limit_mib=500, timeout=60)
print(f"  残存: {left or 'なし'}")
print(f"  GPU : {'解放' if _idle >= 0 else '未解放（60 秒待っても使用中）'}")
save_output("smoke.txt", "leftover=" + (",".join(left) if left else "none") + "\\n"
                         + f"gpu_idle_after={_idle}\\n")
if left:
    fail("smoke 後にプロセスが残存しました: " + ", ".join(left))
if _idle < 0:
    fail("smoke 後に GPU が解放されません（nvidia-smi を確認）")
print("  ✅ smoke 完了（プロセス停止 / GPU 解放を確認）")
'''),
    ]
    pack_nb.write(OUT, cells)
else:
    flow.write(OUT)

# ── kernel-metadata.json（push 用・まだ push しない）─────────
sources = [DATASET_ID, f"{OWNER}/my-cloudflared-bin"]
# wheelhouse 2 点（本体 + cublas patch）。patch が無いと `cuda-toolkit 13.0.3` の
# `nvidia-cublas==13.1.1.3.*` を満たせない（CPU 実測 2026-10-31）。
# 解決に失敗しても install 側が PyPI へフォールバックする。
sources += [f"{OWNER}/vllm-wheelhouse", f"{OWNER}/vllm-wheelhouse-patch"]
meta = {
    "id": f"{OWNER}/{SLUG}",
    # title は slug に解決できる必要がある（不一致だと push が 409）
    "title": "Step2 vLLM qwen38 27b cloudflared",
    "code_file": OUT.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": sources,
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
}
meta_path = OUT_DIR / "kernel-metadata.json"
metadata.write(meta_path, meta)
print(f"wrote {meta_path}")
