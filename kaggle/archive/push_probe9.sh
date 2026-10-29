#!/bin/bash
cd "$(dirname "$0")"
for k in specmat filemat loadflow; do
  for i in $(seq 1 40); do
    out=$(cd "step2-vllm-$k" && kaggle kernels push -p . 2>&1 | tail -1)
    if echo "$out" | grep -q 'successfully pushed'; then
      echo "$k: pushed"; break
    fi
    echo "$k: 待機 ($i) $out"; sleep 60
  done
done
