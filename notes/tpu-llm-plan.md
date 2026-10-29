# TPU で LLM を serving する案（保留・下調べ）

最終更新: 2026-10-29

## 0. 結論（先に・2026-10-29 実測）

- **CLI push では TPU が付かない。UI の Settings → Accelerator が必要。**
- ただし原因は「CLI が古い」だけではない。**CLI 2.2.5（最新）+ `--accelerator TpuV5E8` でも
  CPU イメージで実行された**（下記の 5 パターンを実測）。
- サーバ側で `docker_image` が
  `gcr.io/kaggle-images/python@sha256:17b2fc…` に**再ピン留め**される。
  push リクエストで `docker_image: ""` を明示しても、メタデータから削除しても、
  新規スラッグで初回から TPU 指定しても同じ。
- `tpuQuota.hasEverRun == false`（20h を一度も使えていない）。
- 公式の回避策（`docker_image` を消す / `dockerImageVersionId` を消す）は
  このアカウント・現行バックエンドでは**効かなかった**。
- PR #1219 は 2.2.5 に未収録（コード確認）。ただし収録されても
  `request.docker_image = None` にするだけで、こちらは既に None を送っている。
  → **クライアント側の修正では解決しない**（サーバ側のピン留め）。
- TPU 用 LLM は「準備だけしておく」枠。GPU クォータが尽きたときの退避先として
  設計だけ残す。実行するには UI で TPU を選んで Save & Run All する。

## 1. 実測ログ（プローブ）

### 1.1 CLI の世代

| 環境 | `kaggle --version` | `--accelerator` |
|---|---|---|
| システム Python 3.9 | 1.7.4.5 | 無い |
| `.venv-kaggle`（Python 3.12） | **2.2.5** | あり（`--no-run` も） |

→ Python 3.9 では最新 CLI が入らない（`pip index` も 1.7.4.5 止まり）。
TPU を試すときは `.venv-kaggle/bin/kaggle` を使う。

### 1.2 試した組み合わせ（すべて CPU で実行）

| # | カーネル | machine_shape | 追加条件 | 結果 |
|---|---|---|---|---|
| 1 | tpu-probe | — (`enable_tpu` のみ) | 旧 CLI | CPU |
| 2 | tpu-probe | TpuV5E8 | 旧 CLI | CPU |
| 3 | tpu-probe2（新規） | TpuV5E8 | 2.2.5 / `docker_image_pinning_type: latest` | CPU |
| 4 | tpu-probe2 | TpuV6E8 | 2.2.5 | CPU |
| 5 | tpu-probe2 | TpuV5E8 | `docker_image: ""` 明示 | CPU |
| 6 | tpu-probe3（新規） | TpuV5E8 | 初回から TPU・docker_image 無し | CPU |
| 7 | tpu-probe2 | TpuV5E8 | `enable_internet: false` | CPU |

pull -m のメタデータは毎回:
```
"docker_image": "gcr.io/kaggle-images/python@sha256:17b2fc...",
"machine_shape": "TpuV5E8"
```

### 1.3 旧プローブのログ

```
TPU_NAME  : None
libtpu    : False
accel devs: []
jax       : 0.11.1
devices   : [CpuDevice(id=0)]
torch_xla : インストールされていない
cpu_count : 4
meminfo   : MemTotal: 32869292 kB
```

v1 と v2 で差なし。GPU と同じ `machine_shape` 経路が塞がっている。

## 2. TPU が使えた場合の構成

### 2.1 ハード
| 項目 | 値（TPU v5e-8 の場合） |
|---|---|
| チップ | 8 |
| HBM | 16 GB / チップ = 合計 128 GB |
| bf16 | 対応（fp8 も） |
| INT4/AWQ | **非対応** |

> 注: Kaggle が v3-8 を割り当てる場合も 16 GB × 8 = 128 GB でほぼ同じ。
> 実機の `jax.devices()` で確認する。

### 2.2 モデル
**今の 27B AWQ は TPU では動かない。** `pack-quantized`（compressed-tensors）に
TPU カーネルが無いため。TPU ブランチは別モデルにする。

| 候補 | サイズ(bf16) | 128 GB に収まる | 備考 |
|---|---|---|---|
| Qwen3-8B | 16 GB | 余裕 | 最初の疎通に最適 |
| Qwen3-14B | 28 GB | 余裕 | 品質と速度のバランス |
| Qwen3-32B | 64 GB | ぎりぎり（KV 次第） | max-model-len を絞る |
| 現行 27B の bf16 原本 | 54 GB | 可 | 配布が無ければ不可 |

### 2.3 ソフト
| 経路 | 状態 | 判断 |
|---|---|---|
| **vLLM TPU** | `vllm/platforms/tpu.py` が `tpu_inference` を要求 | 第一候補。ただしモデル対応表を確認 |
| PyTorch/XLA + transformers | `torch_xla` を pip で追加 | 動くが遅い。フォールバック |
| 純 JAX / Flax | `jax` は入っている | 実装コスト大。最終手段 |

vLLM 0.32 の TPU プラグインは外部パッケージ:
```
pip install tpu_inference
```
見つからない場合のログ:
`tpu_inference not found, please install tpu_inference to run vllm on TPU`

### 2.4 再利用できるもの
`auth_proxy.py`（認証・`--enforce-sampling`・`/quota`・`/stop`・`/tmp/keepalive.conf`）、
`llama-chat.html`、cloudflared、keepalive、Gist 配布は **エンジン非依存**。
上流ポートを差し替えるだけでそのまま使える。

```
TPU(vLLM or torch_xla) :8000
        ↑
auth_proxy            :8080  ← UI / 認証 / sampling 矯正 / keepalive リセット
        ↑
cloudflared quick tunnel → 公開 URL
```

## 3. 手順（TPU が付くようになったら）

1. Web UI で Accelerator = TPU に設定 → プローブを再実行
   - `jax.devices()` に `TpuDevice` が出るか
   - `import torch_xla` / `import tpu_inference` の可否
2. `tpu_inference` を導入して `vllm serve` が起動するか
3. モデル: Qwen3-8B bf16 で疎通 → 14B へ
   - 重みは HF Hub から実行時 DL（`enable_internet: true`）
   - ディスク容量に注意（`/kaggle/working` は小さい。`/kaggle/tmp` を使う）
4. 速度計測（tokens/s）と VRAM ならぬ HBM 使用量
5. proxy + cloudflared を被せて 5 点検証（UI / models / streaming / OpenAI 互換 / 会話）
6. keepalive は TPU クォータ 20h に対して長め（例 120 分）でよい

## 4. リスク

- **TPU が CLI から使えない**（現時点の最大のブロッカー）
- モデル配布: 27B の bf16 原本が無い可能性。8B/14B で妥協
- ディスク: bf16 重み + キャッシュで 20〜60 GB。TPU VM の空き要確認
- `tpu_inference` の対応モデルが限られる（Qwen3 系が入っているか要確認）
- TPU VM は起動に数分。keepalive 前提の長時間運用に向く
- `torch_xla` と `torch` のバージョン整合が厳しい（GPU の torch 2.13 とは別環境）

## 5. TODO

- [ ] **UI で TPU を選んでプローブ実行**（現時点で唯一の手段）
      https://www.kaggle.com/code/yamahei21python/tpu-probe3
      Settings → Accelerator → TPU v5e-8 → Save & Run All
- [ ] 成功時の pull -m メタデータを保存（`docker_image` が null/空になるか確認）
- [ ] 失敗時は Kaggle サポートに #1197 の再現として報告
- [ ] `tpu_inference` の対応モデル一覧を確認
- [ ] Qwen3-8B bf16 で `vllm serve` の起動可否
- [ ] TPU 用ビルダー `kaggle/build_step2_tpu.py` を作る（proxy/cloudflared は流用）
- [ ] GPU クォータ枯渇時のフォールバック手順として README に追記

## 6. 参考: machine_shape の値（公式ドキュメント）

```
"NvidiaTeslaT4"  GPU T4 x2
"NvidiaL4"
"TpuV5E8"        TPU v5e-8
```

`kaggle kernels push` は 1.7.4.5 時点でこのフィールドを送らない。
