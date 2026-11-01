# vllm-wheelhouse-patch（Dataset 素材）

`vllm-wheelhouse` で欠けている / 版が合わない wheel だけを置く。Dataset
`yamahei21python/vllm-wheelhouse-patch` の中身。

## なぜ必要か（CPU 実測 2026-10-31）

| 症状 | 原因 |
|---|---|
| `No matching distribution: nvidia-cublas==13.1.1.3.*` | `torch 2.13.0` → `cuda-toolkit 13.0.3.0` が cublas を固定。wheelhouse は 13.8.2.12 |
| `conflict: nvidia-cuda-nvrtc` | 同様に nvrtc 13.0.88 が無い |
| `conflict: nvidia-cuda-runtime` | 同様に cudart 13.0.96 が無い |

本体 wheelhouse は 190 wheel / 3.25 GB あるが、**cuda-toolkit 13.0.3 の pin と
合わない版**が入っている。patch 側で pin 一致の wheel を足す。

## 再取得（Mac / Linux どちらでも可）

```bash
python3 -m pip download --no-deps --only-binary=:all: \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
  --platform manylinux_2_25_x86_64 --platform manylinux2014_x86_64 \
  --platform manylinux2010_x86_64 --python-version 313 \
  -d kaggle/whl-patch \
  "nvidia-cublas==13.1.1.3" "nvidia-cuda-nvrtc==13.0.88" \
  "nvidia-cuda-runtime==13.0.96.*" "nvidia-cuda-cupti==13.0.85.*" \
  "nvidia-cufft==12.0.0.61.*" "nvidia-cufile==1.15.1.6.*" \
  "nvidia-curand==10.4.0.35.*" "nvidia-cusolver==12.0.4.66.*" \
  "nvidia-cusparse==12.6.3.3.*" "nvidia-nvjitlink>=13.0.88,<14" \
  "nvidia-nvtx==13.0.85.*"
```

`--platform` は**複数必要**。cupti は `manylinux_2_25`、cublas は `manylinux_2_27`
とタグが揃っておらず、1 つだけ指定すると「0.0.0a0 しか無い」と誤報する。

## 更新

```bash
kaggle datasets version -p kaggle/whl-patch --dir-mode zip \
  -m "sync cuda-toolkit 13.0.3 pins"
```

`*.whl` は `.gitignore` 済み（404 MB をコミットしない）。
