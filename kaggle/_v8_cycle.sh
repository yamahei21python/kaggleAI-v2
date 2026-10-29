#!/bin/zsh
# v7 完了待ち → v8 push → v8 完了待ち → 成果物回収
# （batch GPU session 上限=2 が T4x2 1セッションで埋まるため、
#   実行中は push が弾かれる。完了を待って入れ替える）
set -u
cd /Users/kohei/Myproject/kaggleAI
SLUG="yamahei21python/step2-orcarouter-qwen38-27b-cloudflared"

st() {
  kaggle kernels status "$SLUG" 2>&1 | sed -n 's/.*status "\(.*\)"/\1/p'
}
sfx() { echo "${1##*.}" }   # KernelWorkerStatus.RUNNING -> RUNNING

echo "=== phase 1: 現行セッション(v7) 完了待ち $(date +%H:%M:%S) ==="
for i in {1..60}; do
  S=$(st); S=${S##*.}
  echo "[p1-$i] ${S:-<empty>} $(date +%H:%M:%S)"
  case "$S" in
    RUNNING|QUEUED) sleep 60 ;;
    COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCELED) break ;;
    *) sleep 30 ;;   # status 取得失敗は完了と判断しない
  esac
done

echo "=== phase 2: v8 push ==="
VER=""
for i in {1..20}; do
  if kaggle kernels push -p kaggle/step2 >/tmp/push_v8.log 2>&1 \
     && grep -q "successfully pushed" /tmp/push_v8.log; then
    VER=$(grep -o 'version [0-9]*' /tmp/push_v8.log | head -1)
    echo "PUSH_OK: ${VER:-version unknown}"
    break
  fi
  echo "retry $i: $(grep -o 'Kernel push error.*' /tmp/push_v8.log | head -1)"
  sleep 30
done
if [ -z "$VER" ]; then
  echo "PUSH_FAILED: v8 未投入。中止（v7 成果物の回収は行わない）"
  exit 1
fi

echo "=== phase 3: v8 完了待ち（新 status を ACTIVE → 終了 の順で見る） ==="
seen_active=0
for i in {1..60}; do
  S=$(st); S=$(sfx "$S")
  echo "[p3-$i] ${S:-<empty>} $(date +%H:%M:%S) active=$seen_active"
  case "$S" in
    RUNNING|QUEUED) seen_active=1; sleep 60 ;;
    COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCELED)
      if (( seen_active )); then break; fi
      echo "  旧版 status の可能性 → 反映待ち"
      sleep 30 ;;
    *) sleep 30 ;;
  esac
done

echo "=== phase 4: 成果物回収 ($S) ==="
if [ -d out ]; then mv out "out.v7.bak.$(date +%H%M%S)"; fi
if ! kaggle kernels output "$SLUG" -p out/ >/tmp/out_v8.log 2>&1; then
  echo "OUTPUT_FAILED: $(tail -3 /tmp/out_v8.log)"
  exit 1
fi
ls out/
if [ ! -s out/verify_result.txt ]; then
  echo "VERIFY_MISSING: verify_result.txt なし。v8 完走と判断するな"
  exit 1
fi
echo "--- verify_result.txt ---"
cat out/verify_result.txt
echo "--- FAILED.txt ---"
cat out/FAILED.txt 2>/dev/null
echo "--- ログの検印ライン ---"
python3 -c '
import json, sys
try:
    d = json.load(open("out/step2-orcarouter-qwen38-27b-cloudflared.log"))
except Exception as e:
    print("log load err:", e); sys.exit()
for e in d:
    if e.get("stream_name") != "stdout":
        continue
    t = e["data"]
    if any(k in t for k in ("キャッシュ復元", "DL が必要", "見つかりません", "Dataset 検出",
                            "symlink", "応答あり", "reply:", "PASS", "FAIL", "PUBLIC URL")):
        print(t, end="")
'
