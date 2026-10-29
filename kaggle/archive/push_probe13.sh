#!/bin/bash
set -e
cd "$(dirname "$0")/cpu-probe13"
echo "=== cpu-probe13 推送開始 ==="
kaggle kernels push -p .
echo "✓ 推送完了"
