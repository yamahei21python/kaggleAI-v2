#!/bin/bash
# CPU セッション上限 5。空きが出るたびに 1 本ずつ push する。
cd "$(dirname "$0")"
for k in flagparse parsersmoke; do
  for i in $(seq 1 40); do
    out=$(cd "step2-vllm-$k" && kaggle kernels push -p . 2>&1 | tail -1)
    if echo "$out" | grep -q 'successfully pushed'; then
      echo "$k: pushed"; break
    fi
    echo "$k: 待機 ($i) $out"; sleep 60
  done
done
