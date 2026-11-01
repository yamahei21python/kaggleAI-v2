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
#   KC_TIMEOUT=900                起動待ちの上限秒
#   KC_NO_OPEN=1                  ブラウザを開かない
#   KC_FORCE_PUSH=1               既に動いていても push し直す
#   KC_TUNNEL=cloudflared         互換のため受けるが、値は 1 つだけ
set -uo pipefail

SCRIPT_DIR="${0:A:h}"
cd "${SCRIPT_DIR}" || exit 1
source "${SCRIPT_DIR}/lib/launcher-common.zsh"
kc_require_kaggle || exit 1

# トンネルは cloudflared の quick tunnel だけ。ngrok は 2026-11-01 に廃止した
# （`notes/url-publishing.md`）。アカウントも authtoken も要らない代わりに
# URL が毎回変わるので、取得はビーコン経由（kc_beacon_url）。
TUNNEL="${KC_TUNNEL:-cloudflared}"
if [[ "${TUNNEL}" != "cloudflared" ]]; then
  echo "KC_TUNNEL=${TUNNEL} は未対応（cloudflared のみ）" >&2
  exit 1
fi
KERNEL_DIR="kaggle/step2"
SLUG="${CFG_KERNEL_SLUG_CLOUDFLARED}"
TIMEOUT="${KC_TIMEOUT:-900}"

# ---- 二重起動の防止 ----
# ダブルクリック 2 回で両方が「未起動」と判断して push すると、
# GPU セッション 2 枠を食ったうえ stop 手段が無い（CLI に無い）ので詰む。
# ロックは起動処理の共通（二重 push で GPU 2 枠を食うのを防ぐ）。
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

open_ui() {  # $1=base url
  local ui_url="$1/"

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

# ---- 既に動いていないか ----
# 固定 URL が無くなっても、ビーコンが「今の URL」を教えてくれるので
# push の前に生存確認ができる（ngrok 版が固定ドメインでやっていたこと）。
# ここを飛ばすと、動いているカーネルの隣で 2 本目を push して GPU 枠を
# 食う（`kaggle kernels stop` は CLI に無いので詰む）。
if [[ "${KC_FORCE_PUSH:-0}" != "1" ]]; then
  BASE="$(kc_beacon_url)"
  if [[ -n "${BASE}" ]] && [[ "$(kc_http_code "${BASE}/healthz")" == "200" ]]; then
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
  BASE=""      # 死んでいた / 古い値。push して取り直す
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

  # URL の出所は 2 つ。ビーコンが本命（Secrets 不要）、Gist は旧経路。
  # 一度取れたら覚えておく（ビーコンは push 直後まだ空のことがある）。
  if [[ -z "${BASE}" ]]; then
    BASE="$(kc_beacon_url)"
    [[ -n "${BASE}" ]] || BASE="$(gist_url)"
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
