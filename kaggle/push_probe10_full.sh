#!/bin/bash
set -e
cd "$(dirname "$0")"

SLUGS="cpu-probe10-compileall cpu-probe10-ldd cpu-probe10-offline cpu-probe10-model"
for slug in $SLUGS; do
  echo "=== $slug ==="
  if [ ! -d "$slug" ]; then
    echo "NOT FOUND: $slug — build_cpu_probe10.py を先に実行"
    continue
  fi
  for i in $(seq 1 40); do
    out=$(cd "$slug" && kaggle kernels push -p . 2>&1 | tail -5)
    if echo "$out" | grep -q 'successfully pushed'; then
      echo "✓ $slug pushed"
      break
    fi
    echo "⇢ retry ($i): $out" | tail -1
    sleep 60
  done
done
echo "ALL DONE"