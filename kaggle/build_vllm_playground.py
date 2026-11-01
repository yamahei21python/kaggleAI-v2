#!/usr/bin/env python3
"""vLLM の**実験用**カーネル（playground）を生成する。

    python3 kaggle/build_vllm_playground.py                     # baseline
    KC_EXPERIMENT=cache python3 kaggle/build_vllm_playground.py
    python3 -m cli push vllm-playground --yes

## ⚠ `cli push` はビルダーを**再実行**する

`cli push` は「古い生成物を本番投入しない」ために push 直前にビルダーを
走らせる（`cli/push.py`）。つまり **`KC_EXPERIMENT` は push のときにも渡す**:

    KC_EXPERIMENT=baseline_long python3 -m cli push vllm-playground --yes

渡さないと既定の `baseline` で上書きされ、**別の実験が走る**。実測 2026-11-02
00:48、`baseline_long` をビルドした直後の push が `baseline`（予算 4,200 秒）を
積んだ。生成ログの 1 行目（`実験: ...`）だけが頼りなので、push 後に確認する。

## なぜ本番と分けるか

これまで実験は `--smoke` で本番カーネルに push していた。smoke 変種は
**同じ kernel id** を持つため、実験のたびに本番のバージョン履歴が書き換わる。
「本当に効果があったものだけを本番へ」を通すには、実験の置き場が別 id で
要る。ここがそれ。

## 本番との違い

| | 本番（`build_step2_vllm_script.py`） | ここ |
|---|---|---|
| id | `step2-vllm-qwen38-27b-<tunnel>-script` | `vllm-playground` |
| トンネル / proxy / keepalive | あり | **なし**（`smoke=True` で省略される） |
| 終わり方 | 公開して待機 | 起動 + 疎通確認で終了 |
| ノブ | 固定 | `KC_EXPERIMENT` で切り替え |

セル（`kaggle/step2-vllm/cells/*.py`）は**本番と同一**を使う。ここだけ別実装に
すると「playground では効いたのに本番では効かない」が起きる。切り替えるのは
環境変数だけにする。

## 実験の一覧

`KC_EXPERIMENT` が選ぶのは**環境変数の初期値だけ**。セルの中身は変えない。
だから playground で当たったノブは、そのまま本番の既定を書き換えれば移せる。
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent          # kaggle/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import kcconfig  # noqa: E402
from kbuild import embed, metadata, pack_script, serve_script  # noqa: E402
from kbuild.settings_cell import render as render_settings  # noqa: E402
from runtime import proxy_source, source_bundle  # noqa: E402
from runtime.constants import PROXY_PORT, VLLM_PORT  # noqa: E402

CFG = kcconfig.load(ROOT)
OWNER = "yamahei21python"
DATASET_SLUG = "qwen38-27b-uncensored-hauhaucs-awq-mtp"
SERVED_NAME = "qwen3.8-27b-awq"
WARM_CACHE_SLUG = "vllm-warm-cache"

CELLS = HERE / "step2-vllm" / "cells"
# 出力先は既定でリポジトリ内。テストが一時ディレクトリへ出すために上書きできる
# （本番の生成物を巻き込まずに「何が書かれるか」を検証するため）。
OUT = pathlib.Path(os.environ.get("KC_PLAYGROUND_OUT")
                   or HERE / "vllm-playground")

# ---- 実験表 ----
# 値は `os.environ.setdefault` で入れる（外から上書きできる）。Kaggle の
# カーネルには実行時 env が無いので、実質ここが唯一のノブ。
#
# どれも**1 変数だけ**動かす。2 つ同時に動かすと、効いたときにどちらが効いたか
# 分からず、v14 の先読み（152.5s のコスト＋重みロードの悪化）と同じ轍を踏む。
EXPERIMENTS = {
    # out2g（10-31, 総 928.7s / 31.7 t/s）の構成を再現する。基準。
    "baseline": (
        "out2g 構成（先読み OFF / キャッシュ復元なし）",
        {"KC_PREWARM": "0", "KC_WARM_CACHE": "0"},
    ),
    # baseline との差が「先読みだけ」になる対照。v14 の再現でもある。
    "prewarm_on": (
        "先読み ON（v14 の再現。重みロード 705s の再現確認）",
        {"KC_PREWARM": "1", "KC_WARM_CACHE": "0"},
    ),
    # 先読み OFF のまま、Triton キャッシュだけ足す。
    # `require` にしてある: Dataset が無い／空のまま走ると **baseline と
    # まったく同じ結果**が出て「キャッシュは効かない」という偽陰性になる。
    # 実測 2026-11-01 時点で `vllm-warm-cache` は未作成なので、この実験は
    # 今は必ず「tar が 1 本も無い」で落ちる（それが正しい）。
    "cache": (
        "先読み OFF + Triton キャッシュ復元（Dataset 必須）",
        {"KC_PREWARM": "0", "KC_WARM_CACHE": "require"},
    ),
    # 373 秒が autotune なのか compile なのかの判定。キャッシュ無しで測る。
    "force_first": (
        "autotune を飛ばす（373 秒の正体の判定）",
        {"KC_PREWARM": "0", "KC_WARM_CACHE": "0",
         "VLLM_TRITON_FORCE_FIRST_CONFIG": "1"},
    ),
    # 構成は `baseline` と同一で、**起動予算だけ** 4,200 → 7,000 秒。
    #
    # なぜ要るか（2026-11-01 22:43 の `cache` 実測）:
    #   復元は成功した（restored=true / entries=236 / 191.1 MiB / 190 秒）が、
    #   試行 1 が重みロードだけで 4,200 秒を使い切り（12:31:13 → 13:39:28、
    #   `layers.34` に到達しただけ）、**コンパイル段に一度も届かなかった**。
    #   つまり `cache` の結果は「効かない」ではなく「未測定」。予算が
    #   「キャッシュが効くか」の交絡因子になっている。
    #   先に「重みロードが遅い」を単独で解かないと、`cache` を何回回しても
    #   同じところで切れる。
    #
    # 2 つの env を同時に動かすのは 2 ノブではなく **1 ノブの裏表**:
    #   試行の上限は `min(ATTEMPT_TIMEOUT, 残予算)` なので、総予算だけ上げて
    #   `ATTEMPT_TIMEOUT` を 4,200 に残すと、試行 1 は**まったく同じ秒数**で
    #   切られる（`03_vllm_serve.py` の `_tmo`）。
    "baseline_long": (
        "baseline と同構成・起動予算だけ 7,000 秒",
        {"KC_PREWARM": "0", "KC_WARM_CACHE": "0",
         "VLLM_STARTUP_BUDGET": "7000", "VLLM_ATTEMPT_TIMEOUT": "7000"},
    ),
}

EXPERIMENT = os.environ.get("KC_EXPERIMENT", "baseline").strip().lower()
if EXPERIMENT not in EXPERIMENTS:
    raise SystemExit(
        f"KC_EXPERIMENT={EXPERIMENT!r} は未対応。"
        f" 使えるのは: {', '.join(sorted(EXPERIMENTS))}")
EXP_DESC, EXP_ENV = EXPERIMENTS[EXPERIMENT]

# 実験 ID を結果に残す。残さないと、あとで `timings.json` を見ても
# どのノブで走った回か分からず、比較表が作れない。
print(f"実験: {EXPERIMENT} — {EXP_DESC}")
print(f"  env: {EXP_ENV or '（既定のまま）'}")

PROXY_SRC_B64 = embed.b64_text(proxy_source(HERE / "step2" / "auth_proxy.py"))
CHAT_SRC_B64 = embed.b64_file(ROOT / "llama-chat.html")

SETTINGS = render_settings(CELLS / "00_settings.py", {
    "__DATASET_SLUG__": DATASET_SLUG,
    "__SERVED__": SERVED_NAME,
    "__TUNNEL__": "none",
    "__PROXY_B64__": PROXY_SRC_B64,
    "__CHAT_B64__": CHAT_SRC_B64,
    "__GIST_ID__": CFG.get("GIST_ID", ""),
    "__GIST_OWNER__": CFG.get("GIST_OWNER", ""),
})

# 実験の環境変数は**設定セルより前**に置く。後ろに置くと、設定セルや install
# セルが既定値で動き出してから書き換えることになり、効かない。
# `import os` を先頭に付けるのは、設定セル側の import より前で使うため。
SETTINGS = (
    "import os\n"
    + "".join(f"os.environ.setdefault({k!r}, {v!r})\n"
              for k, v in EXP_ENV.items())
    + "\n" + SETTINGS
)

ENGINE_CODE = (
    pack_script.wrap_cell(CELLS / "01_install_vllm.py", "def stage_install():\n")
    + "\n\n"
    + pack_script.wrap_cell(
        CELLS / "03_vllm_serve.py", "def launch(model_dir):\n",
        # 起動セルは実測した max_model_len で CONTEXT_LENGTH を上書きする。
        # 関数化するとローカル変数になるため、モジュール側を更新させる。
        tail="\n    return proc\n",
        globals_=("CONTEXT_LENGTH",))
)

ENGINE = {
    "name": "vllm",
    "version": "0.32.0",
    "alias": SERVED_NAME,
    "pattern": "-f vllm",
    "settings": SETTINGS,
    "engine_code": ENGINE_CODE,
    "checks_code": (CELLS / "06_checks.py").read_text(encoding="utf-8"),
    "warmup": pack_script.wrap_cell(CELLS / "04_warmup.py", "def warmup():\n"),
    "work": "/tmp/vllm-serve",
    "flow": {
        "model": False,
        "tp": True,
        "install": True,
        # smoke=True が公開系を丸ごと飛ばすので、ここは本番と同じでよい。
        "publish": True,
        "verify_public": True,
        "keepalive": False,
        "conf": True,
    },
    "data_init": ('{"engine": ENGINE, "stage": "init", "tunnel": TUNNEL_KIND, '
                  '"model": MODEL_NAME, "context_length": CONTEXT_LENGTH, '
                  f'"smoke": SMOKE, "experiment": {EXPERIMENT!r}}}'),
}

SLUG = "vllm-playground"
META = metadata.base(
    id=f"{OWNER}/{SLUG}",
    title=SLUG.replace("-", " "),
    code_file="main.py",
    kernel_type="script",
    gpu=True,
    # 本番と同じモデル + wheelhouse。加えて warm-cache（復元の ON/OFF は
    # `KC_WARM_CACHE` で切り替えるので、付けたまま対照群が取れる）。
    datasets=[f"{OWNER}/{DATASET_SLUG}",
              f"{OWNER}/vllm-wheelhouse",
              f"{OWNER}/vllm-wheelhouse-patch",
              f"{OWNER}/{WARM_CACHE_SLUG}"],
)

if __name__ == "__main__":
    serve_script.write(
        OUT,
        ENGINE,
        META,
        run_minutes=60,
        runtime=source_bundle("net", "proc", "storage", "keepalive", "proxy",
                              "gist"),
        proxy_b64=PROXY_SRC_B64,
        html_b64=CHAT_SRC_B64,
        gist_id=CFG.get("GIST_ID", ""),
        gist_owner=CFG.get("GIST_OWNER", ""),
        port=VLLM_PORT,
        proxy_port=PROXY_PORT,
        hard_max_minutes=180,
        smoke=True,          # トンネル / Gist / keepalive を省略して起動で終了
        tunnel="none",
        builder="kaggle/build_vllm_playground.py",
    )
