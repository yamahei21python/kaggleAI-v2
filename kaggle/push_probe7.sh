#!/bin/bash
# CPU セッション上限 5。空きが出るたびに 1 本ずつ push する。
cd "$(dirname "$0")"
for k in pypiwheels specschema tplflag ioread vram2; do
  for i in $(seq 1 40); do
    if (cd "step2-vllm-$k" && kaggle kernels push -p . 2>&1 | tail -1 | grep -q 'successfully pushed'); then
      echo "$k: pushed"; break
    fi
    echo "$k: 待機 ($i)"; sleep 60
  done
done
