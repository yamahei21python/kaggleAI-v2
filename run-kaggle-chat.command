#!/bin/zsh
# Kaggle で AI を起動して、ブラウザのチャット UI を開く。
#
# ダブルクリック or `./run-kaggle-chat.command` で:
#   1. 既に動いていれば push しない（そのまま再利用）
#   2. 動いていなければ push（GPU T4 x2）
#   3. 公開URL が応答するまで待つ
#   4. ブラウザでチャット UI を開く
#
# 認証は無し（URL を知っていれば誰でも使える）。閉じたいときはノート側で
# 環境変数 API_PASS を設定すると Basic 認証が有効になる。
#
# 環境変数:
#   KC_TUNNEL=ngrok|cloudflared   既定 ngrok（URL 固定・待たされない）
#   KC_DOMAIN=<host>              ngrok の固定ドメインを上書き（既定は config.env）
#   KC_TIMEOUT=900                起動待ちの上限秒
#   KC_NO_OPEN=1                  ブラウザを開かない
#   KC_FORCE_PUSH=1               既に動いていても push し直す
#   KC_RELAY_PORT=19950           ngrok 警告ページ回避用のローカル中継ポート
set -uo pipefail

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR}" || exit 1
source "${SCRIPT_DIR}/lib/launcher-common.zsh"
kc_require_kaggle || exit 1

TUNNEL="${KC_TUNNEL:-ngrok}"
DOMAIN="${KC_DOMAIN:-${CFG_NGROK_DOMAIN}}"
TIMEOUT="${KC_TIMEOUT:-900}"
RELAY_PORT="${KC_RELAY_PORT:-${CFG_RELAY_PORT:-19950}}"

case "${TUNNEL}" in
  ngrok)       KERNEL_DIR="kaggle/step2-ngrok"
               SLUG="${CFG_KERNEL_SLUG_NGROK}" ;;
  cloudflared) KERNEL_DIR="kaggle/step2"
               SLUG="${CFG_KERNEL_SLUG_CLOUDFLARED}" ;;
  *) echo "KC_TUNNEL=${TUNNEL} は未対応（ngrok / cloudflared）" >&2; exit 1 ;;
esac

# ---- 二重起動の防止 ----
# ダブルクリック 2 回で両方が「未起動」と判断して push すると、
# GPU セッション 2 枠を食ったうえ stop 手段が無い（CLI に無い）ので詰む。
# ロックはトンネル共通（ngrok と cloudflared を同時に push しても同じ枠を奪い合う）。
LOCK="/tmp/kc-chat.lock"
if ! mkdir "${LOCK}" 2>/dev/null; then
  # 異常終了（SIGKILL 等）で残ったロックは 30 分で無効とみなす。
  if [[ -n "$(find "${LOCK}" -maxdepth 0 -mmin +30 2>/dev/null)" ]]; then
    echo "古いロックを解除します: ${LOCK}"
    rm -rf "${LOCK}"
    mkdir "${LOCK}" 2>/dev/null || { echo "ロック取得に失敗" >&2; exit 1; }
  else
    echo "別の起動処理が進行中です（${LOCK}）。"
    echo "終わらない場合: rm -rf ${LOCK}"
    exit 1
  fi
fi
# INT/TERM/HUP は exit させる（trap で rmdir だけすると zsh は処理を続行し、
# Ctrl+C を押しても起動待ちループが回り続ける）。後片付けは EXIT trap に一本化。
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# ngrok 無料枠はブラウザ UA の HTML 取得に警告ページを挟む。
# 502/503（proxy 未起動）と混同しないよう、ngrok 固有の痕跡で判定する。
needs_visit() {  # $1=base url
  local ua body
  ua="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
  body="$(curl -s -m 6 -A "${ua}" "$1/" 2>/dev/null | head -c 4000)"
  [[ "${body}" != *"Kaggle Chat"* && "${body}" == *"ngrok"* ]]
}

open_ui() {  # $1=base url
  local ui_url="$1/"
  if [[ "${TUNNEL}" == "ngrok" ]] && needs_visit "$1"; then
    if kc_start_relay "$1" "${RELAY_PORT}"; then
      ui_url="http://127.0.0.1:${RELAY_PORT}/"
      echo ""
      echo "  ℹ ngrok の警告ページを回避するため、ローカル中継を立てました。"
      echo "    中継 PID: ${KC_RELAY_PID}（止めるとき: kill ${KC_RELAY_PID}）"
    fi
  fi

  echo ""
  echo "============================================================"
  echo "  チャット UI : ${ui_url}"
  echo "  API         : $1/v1"
  echo "============================================================"
  if [[ "${KC_NO_OPEN:-0}" != "1" ]]; then
    open "${ui_url}" >/dev/null 2>&1 || true
  fi
}

gist_url() {
  local g u
  g="https://gist.githubusercontent.com/${CFG_GIST_OWNER}/${CFG_GIST_ID}/raw/url.txt"
  u="$(curl -fsS -m 10 "${g}?t=$(date +%s)" 2>/dev/null || true)"
  # Gist は誰でも書き換え可能な前提で扱う。想定ホスト以外は信用しない
  # （前方一致だと `https://a.trycloudflare.com.evil.tld` を通す）。
  if [[ "${u}" =~ '^https://[a-z0-9-]+\.trycloudflare\.com/?$' ]]; then
    printf '%s' "${u%/}"
  fi
}

echo "▶ tunnel=${TUNNEL}"
BASE=""

# ---- ngrok: 固定ドメイン ----
if [[ "${TUNNEL}" == "ngrok" ]]; then
  BASE="https://${DOMAIN}"
  hz="$(kc_http_code "${BASE}/healthz")"
  tags="$(kc_http_code "${BASE}/api/tags")"
  echo "   状態: healthz=${hz} tags=${tags}"

  if [[ "${KC_FORCE_PUSH:-0}" != "1" && "${hz}" == "200" ]]; then
    if kc_model_ready "${BASE}"; then
      echo "✅ 既に起動済み・会話可能（push しない）"
    else
      echo "ℹ 起動済みだがモデル未常駐（ロード中）。push しない。"
      echo "⏳ VRAM 常駐を待つ（最大 ${TIMEOUT}s）..."
      kc_wait_model "${BASE}" "${TIMEOUT}" \
        || echo "⚠ 待ちきれませんでした。UI は開きます（生成が遅いだけの可能性）。"
    fi
    open_ui "${BASE}"
    exit 0
  fi
fi

# ---- push ----
echo "▶ push: ${SLUG}"
kc_push_kernel "${KERNEL_DIR}" || exit 1

# ---- 起動待ち ----
echo "⏳ 起動待ち（最大 ${TIMEOUT}s）..."
deadline=$(( $(date +%s) + TIMEOUT ))
tries=0
while (( $(date +%s) < deadline )); do
  tries=$(( tries + 1 ))

  if [[ "${TUNNEL}" == "cloudflared" ]]; then
    BASE="$(gist_url)"
  fi

  if [[ -n "${BASE}" ]]; then
    # モデルが VRAM に乗るまで待つ。乗る前に開くと UI は出るが、
    # 送信してもロード待ちで数十秒〜数分沈黙する。
    if kc_model_ready "${BASE}"; then
      echo "✅ 会話可能（モデルが VRAM に常駐、${tries} 回目）"
      open_ui "${BASE}"
      exit 0
    fi
    if [[ "$(kc_http_code "${BASE}/api/tags")" == "200" ]]; then
      echo "   ⏳ モデルをロード中（VRAM 未常駐）..."
    fi
  fi

  if (( tries % 6 == 0 )); then
    echo "   ... ${tries} 回 / ${BASE:-URL 未確定}"
  fi
  sleep 10
done

echo "❌ ${TIMEOUT}s 待っても応答しませんでした。" >&2
echo "   Kaggle のログを確認してください: https://www.kaggle.com/code/${SLUG}" >&2
exit 1
