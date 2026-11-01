"""カーネル台帳（1 カーネル = 1 spec）。

id / 出力先 / ビルダー / 環境変数 / datasets をここに集約する。
CLI（`python3 -m cli`）と `tests/test_spec.py` はこの台帳だけを見る。

`status`:
  active  … 常用。リファクタ後も同じ id で push する
  semi    … 必要時のみ（測定・CPU 検証・代替経路）
  retired … 参照しない。ファイルは残す

id は変えない（Secrets / dataset_sources / 実行履歴が id に紐づく）。
"""
import pathlib
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
OWNER = "yamahei21python"

ACTIVE, SEMI, RETIRED = "active", "semi", "retired"


@dataclass(frozen=True)
class Kernel:
    name: str                     # slug（id の後半）
    title: str
    kind: str                     # "notebook" | "script"
    out: str                      # リポジトリ相対の出力ディレクトリ
    code_file: str
    gpu: bool = False
    datasets: tuple = ()
    builder: str = ""             # kaggle/build_*.py のモジュール名（空 = ビルダー無し）
    env: tuple = ()               # ((KEY, VALUE), ...) ビルダーへ渡す環境変数
    metadata_file: str = "kernel-metadata.json"
    status: str = ACTIVE
    smoke_out: str = ""           # smoke 版の出力先（空 = out と同じ場所）

    @property
    def id(self) -> str:
        return f"{OWNER}/{self.name}"

    @property
    def dir(self) -> pathlib.Path:
        return ROOT / self.out

    @property
    def smoke_dir(self) -> pathlib.Path:
        """smoke 版の出力ディレクトリ（通常版を上書きしないための別パス）。"""
        return ROOT / (self.smoke_out or self.out)

    @property
    def metadata_path(self) -> pathlib.Path:
        return self.dir / self.metadata_file


def _k(name, title, kind, out, code, gpu=False, ds=(), builder="", env=(),
       status=ACTIVE, meta="kernel-metadata.json", smoke_out=""):
    return Kernel(name=name, title=title, kind=kind, out=out, code_file=code,
                  gpu=gpu, datasets=tuple(ds), builder=builder,
                  env=tuple(env), status=status, metadata_file=meta,
                  smoke_out=smoke_out)


_K = _k

KERNELS = (
    # ---- step2 / Ollama GGUF ----
    _K("step1-orcarouter-qwen38-27b-dl", "Step1: orcarouter Qwen3.8-27B DL",
       "notebook", "kaggle/step1",
       "step1-orcarouter-qwen38-27b-dl.ipynb", builder="build_step1"),
    _K("step2-orcarouter-qwen38-27b-cloudflared",
       "Step2 orcarouter qwen38 27b cloudflared", "notebook", "kaggle/step2",
       "step2-orcarouter-qwen38-27b-cloudflared.ipynb", gpu=True,
       ds=("yamahei21python/my-ollama-orcarouter-qwen38-27b",
           "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_step2", env=(("STEP2_TUNNEL", "cloudflared"),)),
    # ngrok は 2026-11-01 に廃止（`notes/url-publishing.md`）。理由は 2 つ:
    # `NGROK_TOKEN` が専用アカウントで読めない（`ConnectionError`）ことと、
    # 固定ドメインを維持する理由がビーコンで消えたこと。ファイルは残す
    # （ノートブックは消さない方針）が、再生成も push もしない。
    _K("step2-orcarouter-qwen38-27b-ngrok-fixed",
       "step2 orcarouter qwen38 27b ngrok fixed", "notebook", "kaggle/step2-ngrok",
       "step2-orcarouter-qwen38-27b-ngrok.ipynb", gpu=True,
       status=RETIRED),
    _K("keepalive-test", "keepalive test", "notebook", "kaggle/step2",
       "step2-keepalive-test.ipynb", builder="build_keepalive_test",
       meta="kernel-metadata-keepalive-test.json"),
    _K("keepalive-test-1", "keepalive test 1", "notebook", "kaggle/step2-ka-test",
       "step2-keepalive-test-1.ipynb", status=RETIRED),
    # ---- llama-server 直起動 ----
    # ngrok 廃止（上と同じ理由）。ノートブックは残すが retired。
    _K("llama-server-ngrok", "llama-server ngrok",
       "notebook", "kaggle/llama-server-ngrok", "llama-server-ngrok.ipynb",
       gpu=True, status=RETIRED),
    # ---- vLLM ----
    _K("prepare-vllm-dataset", "Prepare vLLM Dataset", "notebook",
       "kaggle/step2-vllm", "prepare-vllm-dataset.ipynb",
       builder="build_vllm_dataset"),
    _K("step2-vllm-qwen38-27b-cloudflared",
       "Step2 vLLM qwen38 27b cloudflared", "notebook",
       "kaggle/step2-vllm-kernel/cloudflared",
       "step2-vllm-qwen38-27b-cloudflared.ipynb", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin",
           "yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch"),
       builder="build_step2_vllm", env=(("STEP2_TUNNEL", "cloudflared"),),
       smoke_out="kaggle/step2-vllm-kernel/smoke-cloudflared"),
    # ngrok 廃止。ノートブックは残すが retired。
    _K("step2-vllm-qwen38-27b-ngrok", "step2 vllm qwen38 27b ngrok", "notebook",
       "kaggle/step2-vllm-kernel/ngrok", "step2-vllm-qwen38-27b-ngrok.ipynb",
       gpu=True, status=RETIRED),
    _K("step2-vllm-dryrun", "Step2 vLLM dryrun", "notebook",
       "kaggle/step2-vllm-dryrun", "step2-vllm-dryrun.ipynb",
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_vllm_dryrun", status=SEMI),
    # ---- Ollama / script 版（notebook 版と並走。id は別なので notebook は無傷）----
    _K("step2-orcarouter-qwen38-27b-cloudflared-script",
       "step2 orcarouter qwen38 27b cloudflared script", "script",
       "kaggle/step2-script/cloudflared", "main.py", gpu=True,
       ds=("yamahei21python/my-ollama-orcarouter-qwen38-27b",
           "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_step2_script", env=(("STEP2_TUNNEL", "cloudflared"),),
       status=SEMI, smoke_out="kaggle/step2-script/smoke-cloudflared"),
    # ngrok 廃止。`main.py` は残すが retired（再生成しない）。
    _K("step2-orcarouter-qwen38-27b-ngrok-script",
       "step2 orcarouter qwen38 27b ngrok script", "script",
       "kaggle/step2-script/ngrok", "main.py", gpu=True, status=RETIRED),
    # ---- vLLM / script 版（notebook 版と並走。id は別なので notebook は無傷）----
    # wheelhouse 2 点を ds に入れる（2026-10-31）。本体だけでは
    # `cuda-toolkit 13.0.3` が要求する `nvidia-cublas==13.1.1.3.*` を満たせず
    # `--no-index` が失敗する（CPU 実測）。patch（cublas 13.1.1.3 単体）を併せて渡す。
    # 解決に失敗しても install 側が PyPI へフォールバックする。
    _K("step2-vllm-qwen38-27b-cloudflared-script",
       "step2 vllm qwen38 27b cloudflared script", "script",
       "kaggle/step2-vllm-script/cloudflared", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin",
           "yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch"),
       builder="build_step2_vllm_script",
       env=(("STEP2_TUNNEL", "cloudflared"),), status=SEMI,
       smoke_out="kaggle/step2-vllm-script/smoke-cloudflared"),
    # ngrok 廃止。`main.py` は残すが retired（再生成しない）。
    _K("step2-vllm-qwen38-27b-ngrok-script",
       "step2 vllm qwen38 27b ngrok script", "script",
       "kaggle/step2-vllm-script/ngrok", "main.py", gpu=True, status=RETIRED),
    # ---- 実験用 playground ----
    # **本番と同じセル**を、公開系（トンネル / proxy / keepalive）だけ外した
    # 骨組みで動かす。ノブは `KC_EXPERIMENT`（baseline / prewarm_on / cache /
    # force_first）で選び、環境変数の初期値だけを変える。
    #
    # これを作った理由: それまで実験は `--smoke` で本番カーネルへ push して
    # いたが、smoke 変種は**同じ kernel id** なので、実験のたびに本番の
    # バージョン履歴が埋まる。実験の置き場を別 id で持つ。
    #
    # `status=SEMI` にするのは、`cli build`（active のみ）と `check_scripts` に
    # 巻き込まないため。KC_EXPERIMENT は実験ごとに変わるので、生成物を
    # 「ビルダー出力と一致するか」で固定すると実験のたびに落ちる。
    # 代わりに `tests/test_vllm_playground.py` が骨組みを検証する。
    _K("vllm-playground", "vllm playground", "script",
       "kaggle/vllm-playground", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch",
           "yamahei21python/vllm-warm-cache"),
       builder="build_vllm_playground", status=SEMI),
    # ---- 観測系（CPU・無料）----
    # Gist への進捗書き込みが通るかを、GPU を使わずに確かめる。2026-11-01 の
    # vllm-playground は 10/28 から Gist を一度も更新しておらず、その理由が
    # どこにも残っていなかった。起動待ちは 30 分あるので、外から見えないまま
    # 待つと**その 30 分が丸ごと無駄になる**。往復は CPU で 1 分で試せる。
    #
    # `status=SEMI` の理由は playground と同じ（常用ではない）。ただし
    # こちらは環境変数で出力が変わらないので、`check_scripts` に載せても
    # 生成物は安定する。GPU を使わない点だけが違う。
    _K("gist-probe", "gist probe", "script",
       "kaggle/cpu-tests/gist", "main.py", gpu=False,
       ds=(), builder="build_gist_probe", status=SEMI),
    # ---- 公開 URL のビーコン（カーネルのコードを掲示板にする）----
    # **実行しないカーネル**。配信カーネルが `kernels_push` で main.py を
    # 書き換え、読み手は `kernels pull` で URL を読む。Secrets も外部
    # サービスも要らないので、専用アカウント運用と両立する
    # （notes/url-publishing.md）。
    #
    # 2 本あるのは、検証が本番の掲示板を上書きしないため。2026-11-01 に
    # 「古いテスト URL を掴む」事故が出ているので、置き場を分ける。
    _K("serve-url-beacon", "serve-url-beacon", "script",
       "kaggle/serve-url-beacon", "main.py", gpu=False,
       ds=(), builder="build_serve_url_beacon", status=SEMI),
    _K("serve-url-beacon-test", "serve-url-beacon-test", "script",
       "kaggle/cpu-tests/beacon-target", "main.py", gpu=False,
       ds=(), builder="build_serve_url_beacon", status=SEMI),
    # ビーコンの往復を CPU だけで確かめる（GPU 0 秒）。
    _K("beacon-probe", "beacon probe", "script",
       "kaggle/cpu-tests/beacon", "main.py", gpu=False,
       ds=(), builder="build_beacon_probe", status=SEMI),
    # ---- llama-server / script 版（notebook 版と並走）----
    # ngrok 廃止。`main.py` は残すが retired（再生成しない）。
    _K("llama-server-ngrok-script",
       "llama-server ngrok script", "script",
       "kaggle/llama-server-script/ngrok", "main.py", gpu=True, status=RETIRED),
    # ---- 配信しないカーネル / script 版（セル列をそのまま 1 ファイルへ）----
    _K("prepare-vllm-dataset-script",
       "prepare vllm dataset script", "script", "kaggle/step2-vllm/script", "main.py", gpu=False,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_vllm_dataset"), ("KC_BATCH_OUT", "kaggle/step2-vllm/script")),
       status=SEMI),
    _K("step1-orcarouter-qwen38-27b-dl-script",
       "step1 orcarouter qwen38 27b dl script", "script", "kaggle/step1/script", "main.py", gpu=False,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_step1"), ("KC_BATCH_OUT", "kaggle/step1/script")),
       status=SEMI),
    _K("keepalive-test-script",
       "keepalive test script", "script", "kaggle/step2/script", "main.py", gpu=False,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_keepalive_test"), ("KC_BATCH_OUT", "kaggle/step2/script")),
       status=SEMI),
    _K("step2-vllm-dryrun-script",
       "step2 vllm dryrun script", "script", "kaggle/step2-vllm-dryrun/script", "main.py", gpu=False,
       ds=(
           "yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin",
           ),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_vllm_dryrun"), ("KC_BATCH_OUT", "kaggle/step2-vllm-dryrun/script")),
       status=SEMI),
    _K("smoke-orcarouter-qwen38-27b-script",
       "smoke orcarouter qwen38 27b script", "script", "kaggle/smoke/script", "main.py", gpu=True,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_smoke"), ("KC_BATCH_OUT", "kaggle/smoke/script")),
       status=SEMI),
    _K("bench-ctx-vs-speed-orcarouter-qwen3-8-27b-script",
       "bench ctx vs speed orcarouter qwen3 8 27b script", "script", "kaggle/bench/script", "main.py", gpu=True,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_bench"), ("KC_BATCH_OUT", "kaggle/bench/script")),
       status=SEMI),
    # title は Kaggle の上限（50 文字）に収める必要があり、slug は title から作られる。
    # `bench-mtp-spec-decoding-…-script` は 53 文字で作成できなかったため短縮した
    # （notebook 版 `bench-mtp-spec-decoding-orcarouter-qwen3-8-27b` は 48 で収まる）。
    _K("bench-mtp-orcarouter-qwen3-8-27b-script",
       "bench mtp orcarouter qwen3 8 27b script", "script", "kaggle/bench_mtp/script", "main.py", gpu=True,
       ds=(),
       builder="build_batch_script",
       env=(("KC_BATCH", "build_bench_mtp"), ("KC_BATCH_OUT", "kaggle/bench_mtp/script")),
       status=SEMI),
    # ---- EXL3 ----
    _K("exl3-serve", "EXL3 serve", "script", "kaggle/exl3-serve-kernel/serve",
       "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-exl3-4bpw",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_exl3_serve"),
    _K("exl3-serve-dryrun", "EXL3 serve dryrun", "script",
       "kaggle/exl3-serve-kernel/dryrun", "main.py",
       ds=("yamahei21python/my-cloudflared-bin",), builder="build_exl3_serve"),
    _K("exl3-serve-keepalive", "EXL3 serve keepalive", "script",
       "kaggle/exl3-serve-kernel/keepalive", "main.py", builder="build_exl3_serve"),
    _K("exl3-qwen38-gpu-check", "exl3 qwen38 gpu check", "script",
       "kaggle/exl3-gpu-check", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-exl3-4bpw",)),
    _K("exl3-dataset-upload", "exl3 dataset upload", "script",
       "kaggle/exl3-dataset-upload", "main.py"),
    _K("exllamav2-cpu-preflight", "ExLlamaV2 CPU preflight", "script",
       "kaggle/exllama-cpu-preflight", "main.py", status=SEMI),
    # ---- LMDeploy / SGLang（vLLM との比較）----
    # LMDeploy は **この AWQ を読めない**（2026-10-31 実測）。モデルは
    # compressed-tensors / group_size=128 で、lmdeploy 0.18.0 の compressed-tensors
    # 対応は group_size=32 のみ:
    #   ValueError: Unsupported compressed-tensors group_size: expected 32, got 128
    # turbomind も同じ根で OOM する。カーネルは残すが、成功は見込めない。
    _K("step2-lmdeploy-qwen38-27b-cloudflared",
       "Step2 LMDeploy qwen38 27b cloudflared", "script",
       "kaggle/lmdeploy-serve", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_engine_serve", status=SEMI),
    # SGLang は **T4（sm75）では不可**（2026-11-01 実測・4 往復で確定）。
    # install は Rust を入れれば通る（rustup 12.7s + pip 488.4s。cp313 wheel が
    # 無いのは `outlines_core==0.1.26` だけ）。しかしモデルの量子化を解釈する段で:
    #   RuntimeError: ('Quantization scheme is not supported for ', 'the current GPU.
    #                   Min capability: 80. ', 'Current capability: 75.')
    # `CompressedTensorsWNA16.get_min_capability()` が 80 で、T4 は 75。Marlin
    # カーネル経路しか無く、Linear 層に Triton の逃げ道も無い。**カーネルは残すが、
    # 成功は見込めない**（記録として実行できる状態にしておく）。
    _K("step2-sglang-qwen38-27b-cloudflared",
       "Step2 SGLang qwen38 27b cloudflared", "script",
       "kaggle/sglang-serve", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/my-cloudflared-bin"),
       builder="build_engine_serve", status=SEMI),
    # ---- 検証・ベンチ・CPU ----
    _K("llama-server-cpu-check", "llama-server CPU check", "script",
       "kaggle/cpu-check", "main.py",
       ds=("yamahei21python/my-ollama-orcarouter-qwen38-27b",
           "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch")),
    # title は slug に一致させる（`Qwen3.8` と書くと `qwen3-8` になって id と衝突する）。
    _K("smoke-orcarouter-qwen38-27b", "Smoke orcarouter Qwen38 27B",
       "notebook", "kaggle/smoke", "smoke-orcarouter-qwen38-27b.ipynb",
       gpu=True, builder="build_smoke", status=SEMI),
    _K("bench-ctx-vs-speed-orcarouter-qwen3-8-27b",
       "Bench: ctx vs speed orcarouter Qwen3.8-27B", "notebook", "kaggle/bench",
       "bench-ctx-orcarouter-qwen38-27b.ipynb", gpu=True, builder="build_bench",
       status=SEMI),
    _K("bench-mtp-spec-decoding-orcarouter-qwen3-8-27b",
       "Bench: MTP spec decoding orcarouter Qwen3.8-27B", "notebook",
       "kaggle/bench_mtp", "bench-mtp-orcarouter-qwen38-27b.ipynb", gpu=True,
       builder="build_bench_mtp", status=SEMI),
    _K("capture-ollama-patch-v2", "Capture Ollama runtime patch v2", "notebook",
       "kaggle/capture_patch", "capture_patch.ipynb",
       builder="build_capture_patch", status=RETIRED),
    _K("ci-cpu", "ci-cpu", "script", "kaggle/ci-cpu", "main.py",
       builder="build_ci_cpu"),
    # wheelhouse の解決可否を CPU（無料）で調べる。GPU 不使用。
    _K("vllm-whl-probe", "vllm whl probe", "script", "kaggle/whl-probe",
       "main.py",
       ds=("yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch"), status=SEMI),
    # ---- CPU テスト専用（tiny モデル。GPU 0 秒）----
    _K("tiny-llama-probe", "tiny llama probe", "script",
       "kaggle/cpu-tests/tiny-llama", "main.py", status=SEMI),
    _K("tiny-ollama-probe", "tiny ollama probe", "script",
       "kaggle/cpu-tests/tiny-ollama", "main.py", status=SEMI),
    _K("vllm-flag-probe", "vllm flag probe", "script",
       "kaggle/cpu-tests/vllm-flags", "main.py", status=SEMI),
    # install の「展開コスト」を CPU で測る（GPU 0 秒）。実測 2026-11-01:
    # wheelhouse を使った回の方が PyPI より 31 秒遅く、「転送が支配」の前提が
    # 外れた。残る変数は展開の仕方（uv / --no-compile / 解決を飛ばす）だけ。
    _K("vllm-install-bench", "vllm install bench", "script",
       "kaggle/cpu-tests/install-bench", "main.py",
       ds=("yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch"), status=SEMI),
    # 重みロードが固まる原因を GPU 0 秒で切り分ける。v10 は 133 秒で成功、
    # v11/v12 は同じ手順で 840 秒無進捗。疑いは (1) GCSFUSE の速度低下、
    # (2) 読み出しが MemAvailable を削る、(3) ローカル実コピーなら速い。
    _K("vllm-io-bench", "vllm io bench", "script",
       "kaggle/cpu-tests/io-bench", "main.py",
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",),
       status=SEMI),
    # SGLang は `pip install 失敗 (1)` で 297 秒で死んでいた（エンジンは未起動）。
    # 原因は `sglang==0.5.22 → outlines==0.1.11 → outlines_core==0.1.26` の完全固定で、
    # 0.1.26 に cp313 の wheel が無いこと。Rust を入れて通す筋と、固定を回避する筋を
    # 同じカーネルで試す。**GPU 0 秒**（install が通らなければ T4 で賭けない）。
    _K("sglang-install-probe", "sglang install probe", "script",
       "kaggle/cpu-tests/sglang-probe", "main.py", status=SEMI),
    # ---- GPU 高度化テスト（1 エンジン = 1 本。KC_EXP でケース切替）----
    # 実験ごとに分けない: GPU はセッション時間課金で、install + ロードの
    # 固定費（実測 2〜4 分）が毎回乗る。境界は「エンジン / Dataset / ハング隔離」。
    _K("gpu-vllm-start", "gpu vllm start", "script",
       "kaggle/gpu-tests/vllm-start", "main.py", gpu=True,
       ds=("yamahei21python/qwen38-27b-uncensored-hauhaucs-awq-mtp",
           "yamahei21python/vllm-wheelhouse",
           "yamahei21python/vllm-wheelhouse-patch"), status=SEMI),
    _K("gpu-llama-mtp", "gpu llama mtp", "script",
       "kaggle/gpu-tests/llama-mtp", "main.py", gpu=True,
       ds=("yamahei21python/my-ollama-orcarouter-qwen38-27b",
           "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch"),
       status=SEMI),
    _K("gpu-ollama-ctx", "gpu ollama ctx", "script",
       "kaggle/gpu-tests/ollama-ctx", "main.py", gpu=True,
       ds=("yamahei21python/my-ollama-orcarouter-qwen38-27b",
           "yamahei21python/my-ollama-orcarouter-qwen38-27b-patch"),
       status=SEMI),
)


def all_kernels():
    return KERNELS


def active():
    return tuple(k for k in KERNELS if k.status != RETIRED)


def find(name: str) -> Kernel:
    """slug / owner-slug / 出力ディレクトリで引く。曖昧なら拒否する。

    ディレクトリの basename（例 "step2"）は複数カーネルで重複しうるので、
    一致が 1 件のときだけ返す。
    """
    key = name.strip().rstrip("/")
    exact = [k for k in KERNELS if key in (k.name, k.id, k.out)]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise KeyError(f"候補が複数: {key} → " +
                       ", ".join(f"{k.id} ({k.out})" for k in exact))
    base = [k for k in KERNELS if key == k.out.rsplit("/", 1)[-1]]
    if len(base) == 1:
        return base[0]
    if len(base) > 1:
        raise KeyError(f"候補が複数: {key} → " +
                       ", ".join(f"{k.id} ({k.out})" for k in base))
    raise KeyError(f"カーネルが見つからない: {name}")


def builders(targets=None):
    """(ビルダー, env) の重複を除いた実行計画を返す。

    targets 省略時は active のみ（retired は明示指定したときだけ）。
    """
    kernels = active() if targets is None else tuple(find(t) for t in targets)
    plan, seen = [], set()
    for k in kernels:
        if not k.builder:
            continue
        key = (k.builder, k.env)
        if key in seen:
            continue
        seen.add(key)
        plan.append((k.builder, dict(k.env)))
    return plan
