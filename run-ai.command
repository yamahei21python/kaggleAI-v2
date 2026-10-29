#!/bin/zsh
# uzu（ローカル 9B）と Kaggle（Qwen3.8 27B）をまとめて起動する。
#
# ダブルクリックすると、どちらを起動するか聞いてから起動する。
#   1. 両方起動（既定）
#   2. Kaggle だけ
#   3. uzu だけ
#   4. 停止（起動中のものを止める）
#
# 環境変数:
#   KC_TARGET=both|kaggle|uzu|stop   確認ダイアログを出さずに直行
#   KC_NO_OPEN=1                     ブラウザを開かない
#   KC_NO_PROMPT=1                   ダイアログを出さない（KC_TARGET 必須）
#   KC_TIMEOUT=900                   Kaggle 起動待ちの上限秒
#   KC_FORCE_PUSH=1                  Kaggle を push し直す
#   UZU_PORT=9931                    uzu の待受ポート
#   UZU_UI_PORT=8899                 uzu の UI 配信ポート
#   KC_RELAY_PORT=19950              Kaggle 中継ポート
set -uo pipefail

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR}" || exit 1

source "${SCRIPT_DIR}/lib/launcher-common.zsh"

# run-uzu-server.command にも同じ値を渡す（export しないと子は既定値を使う）
export UZU_PORT="${UZU_PORT:-${CFG_UZU_PORT:-9931}}"
export UZU_UI_PORT="${UZU_UI_PORT:-${CFG_UI_PORT:-8899}}"
RELAY_PORT="${KC_RELAY_PORT:-${CFG_RELAY_PORT:-19950}}"
TIMEOUT="${KC_TIMEOUT:-900}"
NO_OPEN="${KC_NO_OPEN:-0}"

# ---- 選択ダイアログ ----------------------------------------
choose_target() {
  # KC_TARGET が明示されていればそれを使う（自動化・再実行用）
  if [[ -n "${KC_TARGET:-}" ]]; then
    echo "${KC_TARGET}"
    return 0
  fi

  # osascript が無い環境（SSH 等）では既定の both にフォールバック
  if ! command -v osascript >/dev/null 2>&1; then
    echo "both"
    return 0
  fi

  local picked
  picked="$(osascript <<'APPLESCRIPT' 2>/dev/null
set opts to {"両方起動（Kaggle + uzu）", "Kaggle だけ（Qwen3.8 27B）", "uzu だけ（ローカル 9B）", "停止（起動中のものを止める）"}
set picked to choose from list opts with title "AI 起動" with prompt "どれを起動しますか？" default items {item 1 of opts} OK button name "起動" cancel button name "キャンセル"
if picked is false then
  return "cancel"
end if
return item 1 of picked
APPLESCRIPT
)"

  case "${picked}" in
    "両方起動"*) echo "both" ;;
    "Kaggle だけ"*) echo "kaggle" ;;
    "uzu だけ"*) echo "uzu" ;;
    "停止"*) echo "stop" ;;
    *) echo "cancel" ;;
  esac
}

TARGET="$(choose_target)"

if [[ "${TARGET}" == "cancel" ]]; then
  echo "キャンセルしました。"
  exit 0
fi

echo "▶ target=${TARGET}"
echo

# ---- 共通ヘルパ --------------------------------------------
port_busy() {  # $1=port
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# uzu は /v1/models が返れば会話可能（モデルロード完了後に応答する）
uzu_ready() {
  [[ "$(kc_http_code "http://127.0.0.1:${UZU_PORT}/v1/models" 3)" == "200" ]]
}

# Kaggle は /api/ps に size_vram > 0 のモデルが居れば会話可能。
# /api/tags は Ollama 起動時点で 200 を返すので判定に使えない。
KAGGLE_DOMAIN="${KC_DOMAIN:-${CFG_NGROK_DOMAIN}}"
KAGGLE_BASE="https://${KAGGLE_DOMAIN}"

# ---- uzu 起動 ----------------------------------------------
# run-uzu-server.command は前面で動き続けるので、バックグラウンドで起動して
# PID を覚えておく。停止時にそれを落とす。
UZU_PID=""
start_uzu() {
  if uzu_ready; then
    echo "✅ uzu: 既に起動済み（port ${UZU_PORT}）"
    return 0
  fi
  if port_busy "${UZU_PORT}"; then
    echo "⚠ uzu: port ${UZU_PORT} は別プロセスが使用中。スキップ。" >&2
    return 1
  fi

  echo "⏳ uzu: 起動中..."
  # ログは /tmp に置く（Finder 起動だとターミナルが閉じても追える）
  UZU_NO_OPEN=1 "${SCRIPT_DIR}/run-uzu-server.command" \
    >/tmp/uzu-launcher.log 2>&1 &
  UZU_PID=$!

  # モデルロード完了を待つ（/v1/models が 200 を返すまで）
  local i
  for i in {1..600}; do
    if uzu_ready; then
      echo "✅ uzu: 会話可能（${i} 回目 / port ${UZU_PORT}）"
      return 0
    fi
    if ! kill -0 "${UZU_PID}" 2>/dev/null; then
      echo "❌ uzu: 起動に失敗（/tmp/uzu-launcher.log）" >&2
      tail -5 /tmp/uzu-launcher.log >&2 2>/dev/null || true
      return 1
    fi
    sleep 1
  done
  echo "⚠ uzu: 600 秒待っても応答なし（/tmp/uzu-launcher.log）" >&2
  return 1
}

# ---- Kaggle 起動 -------------------------------------------
start_kaggle() {
  if kc_model_ready "${KAGGLE_BASE}"; then
    echo "✅ Kaggle: 既に起動済み・会話可能"
  else
    # トンネルは生きているがモデル未常駐なら、push せずに待つ
    if [[ "$(kc_http_code "${KAGGLE_BASE}/healthz" 5)" == "200" ]]; then
      echo "ℹ Kaggle: 起動済みだがモデル未常駐。VRAM 常駐を待つ..."
    else
      echo "⏳ Kaggle: push 中..."
      kc_require_kaggle || return 1
      kc_push_kernel "${SCRIPT_DIR}/kaggle/step2-ngrok" || return 1
    fi
    # モデルが VRAM に乗るまで待つ（実測: ロードに 250 秒前後）
    if ! kc_wait_model "${KAGGLE_BASE}" "${TIMEOUT}"; then
      echo "⚠ Kaggle: ${TIMEOUT}s 待ってもモデル未常駐。UI は開きます（遅い可能性）。" >&2
    fi
  fi

  # ---- 中継を立てる（ngrok の警告ページ回避）----
  if [[ "$(kc_http_code "${KAGGLE_BASE}/healthz" 5)" != "200" ]]; then
    echo "⚠ Kaggle: 上流が応答していません（中継は立てます）" >&2
  fi
  if kc_start_relay "${KAGGLE_BASE}" "${RELAY_PORT}"; then
    echo "✅ Kaggle: 中継起動（PID ${KC_RELAY_PID} / port ${RELAY_PORT}）"
  fi
  return 0
}

# ---- 停止 --------------------------------------------------
stop_all() {
  echo "🛑 起動中のものを停止"

  # Kaggle 中継
  if pkill -f "auth_proxy.py --port ${RELAY_PORT}" 2>/dev/null; then
    echo "  ✅ Kaggle 中継を停止"
  else
    echo "  —  Kaggle 中継は動いていない"
  fi

  # uzu（run-uzu-server.command と cli server の両方）
  if pkill -f "run-uzu-server.command" 2>/dev/null; then
    echo "  ✅ uzu ランチャーを停止"
  fi
  if pkill -f "uzu/target/release/cli server" 2>/dev/null; then
    echo "  ✅ uzu サーバーを停止"
  fi
  # UI は auth_proxy.py が配信している（旧: python -m http.server）
  if pkill -f "auth_proxy.py --port ${UZU_UI_PORT}" 2>/dev/null \
     || pkill -f "http.server ${UZU_UI_PORT}" 2>/dev/null; then
    echo "  ✅ uzu UI サーバーを停止"
  fi

  sleep 1
  local left=()
  port_busy "${UZU_PORT}" && left+=("uzu:${UZU_PORT}")
  port_busy "${UZU_UI_PORT}" && left+=("uzu-ui:${UZU_UI_PORT}")
  port_busy "${RELAY_PORT}" && left+=("relay:${RELAY_PORT}")

  if (( ${#left} > 0 )); then
    echo "  ⚠ 残存: ${(j:, :)left}"
  else
    echo "  ✅ 全停止"
  fi

  echo
  echo "  ℹ Kaggle の GPU セッションはここからは止められません。"
  echo "    止めるには UI から: https://www.kaggle.com/code/${CFG_KERNEL_SLUG_NGROK}"
}

# ---- 本体 --------------------------------------------------
RC=0

case "${TARGET}" in
  stop)
    stop_all
    exit 0
    ;;
  both)
    start_uzu || RC=1
    echo
    start_kaggle || RC=1
    ;;
  kaggle)
    start_kaggle || RC=1
    ;;
  uzu)
    start_uzu || RC=1
    ;;
  *)
    echo "❌ KC_TARGET=${TARGET} は未対応（both / kaggle / uzu / stop）" >&2
    exit 1
    ;;
esac

echo
echo "============================================================"

# UI は auth_proxy.py が / で配る（/llama-chat.html は上流へ転送されて 502 になる）
UZU_URL="http://127.0.0.1:${UZU_UI_PORT}/"
KAGGLE_UI="http://127.0.0.1:${RELAY_PORT}/"

case "${TARGET}" in
  both)
    echo "  uzu    : ${UZU_URL}"
    echo "  Kaggle : ${KAGGLE_UI}"
    echo "  opencode: uzu-local/uzu-stage  |  kaggle-ollama/orcarouter/..."
    ;;
  uzu)
    echo "  uzu    : ${UZU_URL}"
    echo "  opencode: uzu-local/uzu-stage"
    ;;
  kaggle)
    echo "  Kaggle : ${KAGGLE_UI}"
    echo "  opencode: kaggle-ollama/orcarouter/..."
    ;;
esac
echo "============================================================"

if [[ "${NO_OPEN}" != "1" ]]; then
  case "${TARGET}" in
    both)
      open "${UZU_URL}" >/dev/null 2>&1 || true
      open "${KAGGLE_UI}" >/dev/null 2>&1 || true
      ;;
    uzu)
      open "${UZU_URL}" >/dev/null 2>&1 || true
      ;;
    kaggle)
      open "${KAGGLE_UI}" >/dev/null 2>&1 || true
      ;;
  esac
fi

exit "${RC}"

