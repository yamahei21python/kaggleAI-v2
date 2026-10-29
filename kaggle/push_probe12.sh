#!/bin/bash
set -e
cd "$(dirname "$0")/cpu-probe12"
echo "=== cpu-probe12 推送開始 (完全オフライン + wheelhouse 接続) ==="
kaggle kernels push -p .
echo "✓ 推送完了。Kaggle Web/CLI で実行結果を確認してください。"
