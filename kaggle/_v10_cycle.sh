#!/bin/zsh
# v10 push → 完了待ち → 成果物回収（v9 は COMPLETE 済なので session 空き）
set -u
cd /Users/kohei/Myproject/kaggleAI
SLUG="yamahei21python/step2-orcarouter-qwen38-27b-cloudflared"

st() {
  kaggle kernels status "$SLUG" 2>&1 | sed -n 's/.*status "\(.*\)"/\1/p'
}
sfx() { echo "${1##*.}" }

echo "=== v10 push $(date +%H:%M:%S) ==="
VER=""
for i in {1..20}; do
  if kaggle kernels push -p kaggle/step2 >/tmp/push_v10.log 2>&1 \
     && grep -q "successfully pushed" /tmp/push_v10.log; then
    VER=$(grep -o 'version [0-9]*' /tmp/push_v10.log | head -1)
    echo "PUSH_OK: ${VER:-version unknown}"
    break
  fi
  echo "retry $i: $(grep -o 'Kernel push error.*' /tmp/push_v10.log | head -1)"
  sleep 30
done
if [ -z "$VER" ]; then
  echo "PUSH_FAILED: v10 未投入。中止"
  exit 1
fi

echo "=== 完了待ち（新 status を ACTIVE → 終了 の順で見る） ==="
seen_active=0
S=""
for i in {1..70}; do
  S=$(sfx "$(st | tail -1)")   # 複数行返っても末尾だけ取る（改行込みで case 不一致=永久待機を防ぐ）
  echo "[$i] ${S:-<empty>} $(date +%H:%M:%S) active=$seen_active"
  case "$S" in
    RUNNING|QUEUED) seen_active=1; sleep 60 ;;
    COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCELED)
      if (( seen_active )); then break; fi
      echo "  旧版 status の可能性 → 反映待ち"
      sleep 30 ;;
    *) sleep 30 ;;
  esac
done

# ループ脱出が上限到達でも「完了」とはしない。新版を一度も見ていないまま
# 回収すると旧版成果物を拾って偽成功する。
case "$S" in
  COMPLETE|ERROR|CANCEL_ACKNOWLEDGED|CANCELED)
    if (( ! seen_active )); then
      echo "TIMEOUT: 新版 status 未観測 ($S)。旧版成果物の可能性 → 中止"
      exit 1
    fi ;;
  *)
    echo "TIMEOUT: status=${S:-<empty>} のまま。中止"
    exit 1 ;;
esac

echo "=== 成果物回収 ($S) ==="
if [ -d out.v10 ]; then mv out.v10 "out.v10.bak.$(date +%H%M%S)"; fi
if ! kaggle kernels output "$SLUG" -p out.v10/ >/tmp/out_v10.log 2>&1; then
  echo "OUTPUT_FAILED: $(tail -3 /tmp/out_v10.log)"
  exit 1
fi
ls out.v10/
if [ ! -s out.v10/verify_result.txt ]; then
  echo "VERIFY_MISSING: verify_result.txt なし。成功と判断するな"
  exit 1
fi
# verify 保存後に fail() → FAILED.txt + raise する経路がある（v8 実績）。
# verify_result.txt の存在だけを成功条件にしない。
if [ "$S" = ERROR ] || [ -s out.v10/FAILED.txt ]; then
  echo "FAILED: status=$S"
  head -3 out.v10/FAILED.txt 2>/dev/null
  exit 1
fi
echo "--- verify_result.txt ---"
cat out.v10/verify_result.txt
echo "--- FAILED.txt ---"
cat out.v10/FAILED.txt 2>/dev/null
echo "--- 検印ライン ---"
python3 -c '
import json, sys
try:
    d = json.load(open("out.v10/step2-orcarouter-qwen38-27b-cloudflared.log"))
except Exception as e:
    print("log load err:", e); sys.exit()
for e in d:
    if e.get("stream_name") != "stdout":
        continue
    t = e["data"]
    if any(k in t for k in ("全数充足", "欠落", "show 失敗", "pull", "blobs ",
                            "ps :", "GPU 常駐", "PASS", "FAIL", "5/5",
                            "応答あり", "速度", "download")):
        print(t, end="")
'
