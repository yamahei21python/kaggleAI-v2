#!/bin/zsh
# ============================================================
# llama-server をローカルで起動する（Ollama を経由しない）
#
# 使い方:
#   1) このファイルをダブルクリック
#   2) ブラウザが自動で開く → そのままチャットできる
#   3) ターミナルで Ctrl+C を押すと停止
#
# 環境変数で上書きできます:
#   LLAMA_MODEL=27b ./run-llama-server.command   # 27B を選ぶ（RAM 16 GiB では動きません）
#   LLAMA_CTX=32768  ./run-llama-server.command  # コンテキスト長を変える
#   LLAMA_PORT=9931  ./run-llama-server.command  # ポートを変える
#   LLAMA_MTP=0      ./run-llama-server.command  # MTP を切る（既定は有効）
#   LLAMA_OPEN=opencode ./run-llama-server.command  # 起動後に opencode を開く
#   LLAMA_UI_PORT=8899 ./run-llama-server.command   # チャット UI 配信ポート
# ============================================================
set -u

SCRIPT_DIR="${0:A:h}"
source "${SCRIPT_DIR}/lib/launcher-common.zsh"

# ---- 設定 ----------------------------------------------------
# モデル: 9b（既定・6.6 GB）または 27b（16.8 GB・RAM 16 GiB では不可）
MODEL_CHOICE="${LLAMA_MODEL:-9b}"

case "$MODEL_CHOICE" in
  9b|9B)
    # huihui_ai/qwen3.5-abliterated:9b の model 層
    MODEL_SHA="afb54ad43a39f947407f5cabc59856348d70e072baa5c62d436332157c151bcd"
    ALIAS="qwen35-9b"
    PULL_HINT="huihui_ai/qwen3.5-abliterated:9b"
    DEFAULT_CTX=32768
    ;;
  27b|27B)
    # orcarouter/Qwen3.8-27B-Uncensored:latest の model 層
    MODEL_SHA="3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec"
    ALIAS="qwen38-27b"
    PULL_HINT="orcarouter/Qwen3.8-27B-Uncensored:latest"
    DEFAULT_CTX=32768
    ;;
  *)
    echo "❌ LLAMA_MODEL は 9b か 27b を指定してください（指定値: $MODEL_CHOICE）"
    kc_pause
    exit 1
    ;;
esac

CTX="${LLAMA_CTX:-$DEFAULT_CTX}"
PORT="${LLAMA_PORT:-9931}"
HOST="127.0.0.1"
USE_MTP="${LLAMA_MTP:-1}"        # 1 で MTP 有効（実測 1.18x）
OPEN_AFTER="${LLAMA_OPEN:-browser}"  # browser | opencode | none
UI_PORT="${LLAMA_UI_PORT:-8899}"
UI_PID=""
SERVER_PID=""

# UI ポートを API ポートと重ねない（中継が上流に向かい詰む）
if [[ "${UI_PORT}" == "${PORT}" ]]; then
  echo "❌ LLAMA_UI_PORT (${UI_PORT}) と LLAMA_PORT (${PORT}) は同じにできません"
  kc_pause
  exit 1
fi

# ---- 事前チェック --------------------------------------------
echo "============================================================"
echo "  llama-server 起動"
echo "============================================================"
echo

if ! LLAMA_SERVER="$(kc_ollama_server_bin)"; then
  echo "❌ llama-server が見つかりません:"
  echo "   /Applications/Ollama.app/Contents/Resources/llama-server"
  echo
  echo "   Ollama.app がインストールされているか確認してください。"
  kc_pause
  exit 1
fi

if ! MODEL_BLOB="$(kc_ollama_blob "${MODEL_SHA}")"; then
  echo "❌ モデルが見つかりません:"
  echo "   ${HOME}/.ollama/models/blobs/sha256-${MODEL_SHA}"
  echo
  echo "   先に Ollama で pull してください:"
  echo "     ollama pull ${PULL_HINT}"
  kc_pause
  exit 1
fi

# 27B は RAM 16 GiB に載らないので警告
if [[ "${MODEL_CHOICE}" == "27b" || "${MODEL_CHOICE}" == "27B" ]]; then
  MEM_GIB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  if (( MEM_GIB < 32 )); then
    echo "⚠  この Mac の RAM は ${MEM_GIB} GiB です。"
    echo "   27B（16.8 GB）は RAM を超えるため、実用速度になりません（実測 0.10 tokens/s）。"
    echo "   続行しますか？"
    kc_ask "   y で続行 / それ以外で中止: "
    [[ "${KC_ANSWER}" == "y" ]] || { echo "中止しました。"; exit 0; }
    echo
  fi
fi

# ポートが使用中なら止める
if kc_port_in_use "${PORT}"; then
  echo "⚠  ポート ${PORT} は使用中です:"
  kc_port_listener "${PORT}" | sed -n '2,4p'
  echo "   このプロセスを停止しますか？"
  kc_ask "   y で停止 / それ以外で中止: "
  if [[ "${KC_ANSWER}" == "y" ]]; then
    kc_kill_port "${PORT}"
    sleep 2
    echo "   ✅ 停止しました"
  else
    echo "中止しました。"
    exit 0
  fi
  echo
fi

# ---- 起動 ----------------------------------------------------
SIZE_GIB="$(awk -v bytes="$(stat -f%z "${MODEL_BLOB}")" 'BEGIN { printf "%.2f", bytes / 1073741824 }')"
echo "  モデル     : ${ALIAS}"
echo "  実体       : $(basename "${MODEL_BLOB}")"
echo "  サイズ     : ${SIZE_GIB} GiB"
echo "  コンテキスト: ${CTX}"
if [[ "${USE_MTP}" == "1" ]]; then
  echo "  MTP        : 有効（--spec-type draft-mtp）"
else
  echo "  MTP        : 無効"
fi
echo "  URL        : http://${HOST}:${PORT}"
echo
echo "  ⏳ ロード中...（初回は数秒〜1分）"
echo

# MTP は投機的デコード。9B で実測 16.0 → 18.9 tokens/s（1.18x）
SPEC_ARGS=()
[[ "${USE_MTP}" == "1" ]] && SPEC_ARGS=(--spec-type draft-mtp)

# KV キャッシュは q8_0 を明示する。
# 指定しないと f16 になり、長い ctx で VRAM を溢れて CPU 推論に落ちる（実測 0.8 tps）
KV_ARGS=(--cache-type-k q8_0 --cache-type-v q8_0)

"${LLAMA_SERVER}" \
  -m "${MODEL_BLOB}" \
  -c "${CTX}" \
  "${KV_ARGS[@]}" \
  -ngl all \
  -fa on \
  -np 1 \
  --host "${HOST}" \
  --port "${PORT}" \
  --jinja \
  --temp 0 \
  --alias "${ALIAS}" \
  --reasoning off \
  --reasoning-budget 0 \
  --cors-origins '*' \
  "${SPEC_ARGS[@]}" \
  > /tmp/llama-server.log 2>&1 &

SERVER_PID=$!

# Ctrl+C / 終了 / ブラウザの停止ボタンで、サーバーと UI 中継を確実に停止
# （--stop-pid で自分を SIGTERM され、ここが走って子プロセスを殺す）
CLEANED=0
cleanup() {
  (( CLEANED )) && return 0
  CLEANED=1
  echo; echo "  停止中..."
  # KC_UI_PID は kc_start_ui_proxy が起動直後に設定する。起動待ちの途中で
  # 止められても中継が孤児にならないよう、呼び出し元の UI_PID と両方見る。
  [[ -n "${UI_PID}" ]] && kill -9 "${UI_PID}" 2>/dev/null
  [[ -n "${KC_UI_PID:-}" ]] && kill -9 "${KC_UI_PID}" 2>/dev/null
  [[ -n "${SERVER_PID}" ]] && kill -9 "${SERVER_PID}" 2>/dev/null
  echo "  ✅ llama-server を停止しました"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---- 起動待ち → ブラウザを開く -------------------------------
READY=0
for i in {1..180}; do
  if curl -s --max-time 2 "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '"ok"'; then
    READY=1
    echo "  ✅ 起動完了（${i}秒）"
    break
  fi
  # プロセスが死んでいないか
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "  ❌ llama-server が落ちました。ログ:"
    echo
    tail -30 /tmp/llama-server.log
    kc_pause
    exit 1
  fi
  sleep 1
done

if (( READY != 1 )); then
  echo "  ❌ 起動がタイムアウトしました（180秒）。ログ:"
  echo
  tail -30 /tmp/llama-server.log
  kill -9 "${SERVER_PID}" 2>/dev/null
  kc_pause
  exit 1
fi

# ---- チャット UI を auth_proxy.py で配信 ----------------------
# file:// だと /quota・/stop が無く「■ 停止」ボタンが出ない。
# 上流は llama-server、自分は --stop-pid に渡す（ランチャーを SIGTERM
# で落とせば cleanup trap が llama-server と UI をまとめて落とす）。
UI_URL="file://${SCRIPT_DIR}/llama-chat.html?port=${PORT}"
kc_start_ui_proxy "http://${HOST}:${PORT}" "${UI_PORT}" $$ "llama" 0
case $? in
  0)
    UI_PID="${KC_UI_PID}"
    UI_URL="${KC_UI_URL}"
    ;;
  2)
    echo "⚠ auth_proxy.py が見つからないため file:// で開きます（停止ボタン無し）:"
    echo "   ${KC_PROXY_PY}"
    ;;
  3)
    echo "❌ UI ポート ${UI_PORT} は使用中です（他のランチャー？）。"
    echo "   LLAMA_UI_PORT=<別ポート> で起動し直してください。"
    kc_pause
    exit 1
    ;;
  *)
    echo "⚠ UI 中継が起動しなかった（/tmp/llama-ui-proxy.log）。file:// で開きます（停止ボタン無し）。"
    ;;
esac

echo
echo "============================================================"
echo "  ✅ 使えます"
echo "============================================================"
echo
echo "  チャット UI : ${UI_URL}"
echo "  OpenAI互換  : http://${HOST}:${PORT}/v1"
echo "  モデル名    : ${ALIAS}"
echo
echo "  Python から:"
echo "    from openai import OpenAI"
echo "    c = OpenAI(base_url=\"http://${HOST}:${PORT}/v1\", api_key=\"dummy\")"
echo "    c.chat.completions.create(model=\"${ALIAS}\", messages=[{\"role\":\"user\",\"content\":\"hi\"}])"
echo
echo "  停止するには Ctrl+C を押してください。"
echo

# ---- 起動後の動作 --------------------------------------------
case "${OPEN_AFTER}" in
  opencode)
    if command -v opencode >/dev/null 2>&1; then
      echo "  🚀 opencode を起動します（モデル: llama-local/${ALIAS}）"
      echo "     ※ 初回は opencode 側にプロバイダ登録が必要です"
      echo
      opencode --model "llama-local/${ALIAS}" "${SCRIPT_DIR}"
    else
      echo "  ⚠  opencode が見つかりません。ブラウザを開きます。"
      open "${UI_URL}"
    fi
    ;;
  none)
    echo "  （ブラウザは開きません。上の URL を手動で開いてください）"
    ;;
  *)
    # 既定: 自前のチャット UI を開く（llama-server 同梱の Web UI は Ollama 版に含まれていない）
    open "${UI_URL}"
    ;;
esac

# ---- 終了まで待つ --------------------------------------------
wait "${SERVER_PID}"
