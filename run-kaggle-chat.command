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
#   KC_DOMAIN=<host>              ngrok の固定ドメインを上書き
#   KC_TIMEOUT=900                起動待ちの上限秒
#   KC_NO_OPEN=1                  ブラウザを開かない
#   KC_FORCE_PUSH=1               既に動いていても push し直す
set -uo pipefail

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR}" || exit 1

# Finder から起動すると PATH が最小限（/usr/bin:/bin:/usr/sbin:/sbin）になる。
# kaggle CLI は pip --user で ~/Library/Python/3.9/bin に入っているので、
# ここを明示しないと「kaggle: command not found」で即死する（実測）。
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${HOME}/Library/Python/3.9/bin:${PATH}"

# それでも見つからない場合は、python3 -m kaggle にフォールバックする。
if ! command -v kaggle >/dev/null 2>&1; then
  if python3 -c "import kaggle" >/dev/null 2>&1; then
    kaggle() { python3 -m kaggle.cli "$@"; }
  else
    echo "❌ kaggle CLI が見つかりません。" >&2
    echo "   pip3 install --user kaggle  を実行してください。" >&2
    exit 1
  fi
fi

TUNNEL="${KC_TUNNEL:-ngrok}"
DOMAIN="${KC_DOMAIN:-rebalance-cornflake-sagging.ngrok-free.dev}"
TIMEOUT="${KC_TIMEOUT:-900}"

case "${TUNNEL}" in
  ngrok)       KERNEL_DIR="kaggle/step2-ngrok"
               SLUG="yamahei21python/step2-orcarouter-qwen38-27b-ngrok-fixed" ;;
  cloudflared) KERNEL_DIR="kaggle/step2"
               SLUG="yamahei21python/step2-orcarouter-qwen38-27b-cloudflared" ;;
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
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT INT TERM HUP

# ---- 状態判定 ----
# healthz は proxy の生存、tags は上流（モデル）の準備完了。
code_of() {  # $1=url
  local c
  c="$(curl -s -o /dev/null -w '%{http_code}' -m 8 \
        -H "ngrok-skip-browser-warning: true" "$1" 2>/dev/null)"
  echo "${c:-000}"
}

# /api/tags は Ollama が起動した時点で 200 を返す。モデルのロード完了は
# 待たない（実測: tags=200 でも 250 秒のロード中だった）。
# 会話できるかは /api/ps に size_vram > 0 のモデルが居るかで判定する。
model_ready() {  # $1=base url
  local body
  body="$(curl -s -m 10 -H "ngrok-skip-browser-warning: true" \
            "$1/api/ps" 2>/dev/null)"
  [[ -z "${body}" ]] && return 1
  # "size_vram":0 または models が空なら未準備。
  # jq 非依存で済むよう grep で見る（値が 0 以外なら VRAM 常駐）。
  [[ "${body}" == *'"models":[]'* ]] && return 1
  [[ "${body}" =~ '"size_vram":[1-9]' ]] && return 0
  return 1
}

# ngrok 無料枠はブラウザ UA の HTML 取得に警告ページを挟む。
# 一度「Visit Site」を押すと cookie が入り、以後は素通りする。
# 502/503（proxy 未起動）と混同しないよう、ngrok 固有の痕跡で判定する。
needs_visit() {  # $1=base url
  local ua body
  ua="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
  body="$(curl -s -m 6 -A "${ua}" "$1/" 2>/dev/null | head -c 4000)"
  [[ "${body}" != *"Kaggle Chat"* && "${body}" == *"ngrok"* ]]
}

open_ui() {  # $1=base url
  local ui_url="$1/"

  # ngrok 無料枠は、ブラウザの HTML 取得に ERR_NGROK_6024 の
  # インタースティシャル（ボタンもリンクも無いブロックページ）を返す。
  # ヘッダ ngrok-skip-browser-warning を付けたリクエストは素通りするので、
  # ローカルに中継を立ててそのヘッダを肩代わりさせる。
  # 中継は UI も配るので、ブラウザからは 127.0.0.1 を開けばよい。
  if [[ "${TUNNEL}" == "ngrok" ]] && needs_visit "$1"; then
    local relay_port=19950
    local proxy_py="${SCRIPT_DIR}/kaggle/step2/auth_proxy.py"
    local ui_tmp
    ui_tmp="$(mktemp -t kcui)"

    # 上流から UI を取得（ヘッダ付きなら通る）
    if curl -fsS -m 20 -H "ngrok-skip-browser-warning: true" \
         -o "${ui_tmp}" "$1/" 2>/dev/null \
       && grep -q "Kaggle Chat" "${ui_tmp}" 2>/dev/null; then
      # 既存の中継を落としてから立てる
      pkill -f "auth_proxy.py --port ${relay_port}" 2>/dev/null || true
      sleep 1
      python3 "${proxy_py}" --port "${relay_port}" --upstream "$1" \
        --chat-html "${ui_tmp}" \
        --upstream-header "ngrok-skip-browser-warning=true" \
        >/tmp/kc-relay.log 2>&1 &
      local relay_pid=$!
      sleep 2
      if kill -0 "${relay_pid}" 2>/dev/null; then
        ui_url="http://127.0.0.1:${relay_port}/"
        echo ""
        echo "  ℹ ngrok の警告ページを回避するため、ローカル中継を立てました。"
        echo "    中継 PID: ${relay_pid}（止めるとき: kill ${relay_pid}）"
      else
        echo "  ⚠ 中継の起動に失敗しました（/tmp/kc-relay.log）" >&2
      fi
    else
      echo "  ⚠ UI の取得に失敗しました（上流が応答していない可能性）" >&2
    fi
    rm -f "${ui_tmp}"
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
  g="https://gist.githubusercontent.com/yamahei21python/2b44ac8cb115012a31d3c46661af933f/raw/url.txt"
  u="$(curl -fsS -m 10 "${g}?t=$(date +%s)" 2>/dev/null || true)"
  # Gist は誰でも書き換え可能な前提で扱う。想定ホスト以外は信用しない
  # （前方一致だと `https://a.trycloudflare.com.evil.tld` を通す）。
  if [[ "${u}" =~ '^https://[a-z0-9-]+\.trycloudflare\.com/?$' ]]; then
    printf '%s' "${u%/}"
  fi
}

echo "▶ tunnel=${TUNNEL}"

# ---- ngrok: 固定ドメイン ----
if [[ "${TUNNEL}" == "ngrok" ]]; then
  BASE="https://${DOMAIN}"
  hz="$(code_of "${BASE}/healthz")"
  tags="$(code_of "${BASE}/api/tags")"
  echo "   状態: healthz=${hz} tags=${tags}"

  if [[ "${KC_FORCE_PUSH:-0}" != "1" ]]; then
    if [[ "${hz}" == "200" ]] && model_ready "${BASE}"; then
      echo "✅ 既に起動済み・会話可能（push しない）"
      open_ui "${BASE}"
      exit 0
    fi
    if [[ "${hz}" == "200" ]]; then
      echo "ℹ 起動済みだがモデル未常駐（ロード中）。push しない。"
      echo "⏳ VRAM 常駐を待つ（最大 ${TIMEOUT}s）..."
      deadline=$(( $(date +%s) + TIMEOUT ))
      while (( $(date +%s) < deadline )); do
        if model_ready "${BASE}"; then
          echo "✅ 会話可能"
          open_ui "${BASE}"
          exit 0
        fi
        sleep 10
      done
      echo "⚠ 待ちきれませんでした。UI は開きます（生成が遅いだけの可能性）。"
      open_ui "${BASE}"
      exit 0
    fi
  fi
fi

# ---- push ----
echo "▶ push: ${SLUG}"
PUSH_OUT="$(kaggle kernels push -p "${KERNEL_DIR}" 2>&1)"
PUSH_RC=$?
echo "${PUSH_OUT}" | sed -n '1,6p'
if [[ "${PUSH_RC}" -ne 0 ]]; then
  if echo "${PUSH_OUT}" | grep -q "Maximum batch GPU session count"; then
    cat >&2 <<'EOS'
❌ GPU セッションが 2 本埋まっています。
   Kaggle の UI で古いカーネルを停止するか、終了を待ってください
   （`kaggle kernels stop` は CLI に存在しません）。
EOS
  else
    echo "❌ push に失敗しました（rc=${PUSH_RC}）。上の出力を確認してください。" >&2
  fi
  exit 1
fi

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
    if model_ready "${BASE}"; then
      echo "✅ 会話可能（モデルが VRAM に常駐、${tries} 回目）"
      open_ui "${BASE}"
      exit 0
    fi
    if [[ "$(code_of "${BASE}/api/tags")" == "200" ]]; then
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
