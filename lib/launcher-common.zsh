# run-kaggle-chat.command / run-ai.command / run-llama-server.command /
# run-uzu-server.command の共通処理。
# 使い方: SCRIPT_DIR を決めてから `source "${SCRIPT_DIR}/lib/launcher-common.zsh"`

# Finder から起動すると PATH が最小限（/usr/bin:/bin:/usr/sbin:/sbin）になる。
# kaggle CLI は pip --user で ~/Library/Python/3.9/bin に入っているので、
# ここを明示しないと「kaggle: command not found」で即死する（実測）。
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${HOME}/Library/Python/3.9/bin:${PATH}"

# ---- 設定（config.env → config.local.env の順に読み、CFG_<KEY> に入れる）----
# 環境変数（KC_DOMAIN など）を潰さないよう、読み込み先は CFG_ 接頭辞に限定する。
kc_load_config() {
  local file key value
  for file in "${SCRIPT_DIR}/config.env" "${SCRIPT_DIR}/config.local.env"; do
    [[ -f "${file}" ]] || continue
    while IFS='=' read -r key value || [[ -n "${key}" ]]; do
      key="${key//[[:space:]]/}"
      [[ -z "${key}" || "${key}" == \#* ]] && continue
      [[ "${key}" =~ '^[A-Z0-9_]+$' ]] || continue
      typeset -g "CFG_${key}=${value}"
    done < "${file}"
  done
}
kc_load_config

KC_UA_HEADER="ngrok-skip-browser-warning: true"
KC_PROXY_PY="${SCRIPT_DIR}/kaggle/step2/auth_proxy.py"
KC_CHAT_HTML="${SCRIPT_DIR}/llama-chat.html"

# kaggle CLI が PATH に無ければ python3 -m kaggle にフォールバックする。
kc_require_kaggle() {
  command -v kaggle >/dev/null 2>&1 && return 0
  if python3 -c "import kaggle" >/dev/null 2>&1; then
    kaggle() { python3 -m kaggle.cli "$@"; }
    return 0
  fi
  echo "❌ kaggle CLI が見つかりません。" >&2
  echo "   pip3 install --user kaggle  を実行してください。" >&2
  return 1
}

kc_http_code() {  # $1=url $2=timeout(既定8)
  local c
  c="$(curl -s -o /dev/null -w '%{http_code}' -m "${2:-8}" \
        -H "${KC_UA_HEADER}" "$1" 2>/dev/null)"
  echo "${c:-000}"
}

# /api/tags は Ollama が起動した時点で 200 を返す。モデルのロード完了は
# 待たない（実測: tags=200 でも 250 秒のロード中だった）。
# 会話できるかは /api/ps に size_vram > 0 のモデルが居るかで判定する。
kc_model_ready() {  # $1=base url
  local body
  body="$(curl -s -m 10 -H "${KC_UA_HEADER}" "$1/api/ps" 2>/dev/null)"
  [[ -z "${body}" ]] && return 1
  [[ "${body}" == *'"models":[]'* ]] && return 1
  [[ "${body}" =~ '"size_vram":[1-9]' ]] && return 0
  return 1
}

# モデルが VRAM に乗るまで待つ。$1=base url $2=timeout 秒
kc_wait_model() {
  local base="$1" timeout="$2" deadline i=0
  deadline=$(( $(date +%s) + timeout ))
  while (( $(date +%s) < deadline )); do
    i=$(( i + 1 ))
    if kc_model_ready "${base}"; then
      echo "✅ 会話可能（モデルが VRAM に常駐、${i} 回目）"
      return 0
    fi
    if (( i % 6 == 0 )); then
      echo "   ⏳ モデルをロード中（VRAM 未常駐）... ${i}"
    fi
    sleep 10
  done
  return 1
}

# kaggle kernels push。GPU セッション上限のときは分かりやすく案内する。
kc_push_kernel() {  # $1=kernel dir
  local out rc
  out="$(kaggle kernels push -p "$1" 2>&1)"
  rc=$?
  echo "${out}" | sed -n '1,6p'
  (( rc == 0 )) && return 0
  if echo "${out}" | grep -q "Maximum batch GPU session count"; then
    cat >&2 <<'EOS'
❌ GPU セッションが 2 本埋まっています。
   Kaggle の UI で古いカーネルを停止するか、終了を待ってください
   （`kaggle kernels stop` は CLI に存在しません）。
EOS
  else
    echo "❌ push に失敗しました（rc=${rc}）。上の出力を確認してください。" >&2
  fi
  return 1
}

# ngrok 無料枠はブラウザの HTML 取得に ERR_NGROK_6024 の警告ページを返す。
# ヘッダ ngrok-skip-browser-warning を付けたリクエストは素通りするので、
# ローカル中継にそのヘッダを肩代わりさせる。UI はローカルの llama-chat.html
# を配る（上流から取ると push 済みの古い版が返ってくる）。
# 成功時は KC_RELAY_PID に PID を入れて 0 を返す。
kc_start_relay() {  # $1=upstream base url $2=port
  KC_RELAY_PID=""
  if [[ ! -f "${KC_PROXY_PY}" || ! -f "${KC_CHAT_HTML}" ]]; then
    echo "⚠ 中継に必要なファイルがありません: ${KC_PROXY_PY} / ${KC_CHAT_HTML}" >&2
    return 1
  fi
  pkill -f "auth_proxy.py --port $2" 2>/dev/null || true
  sleep 1
  python3 "${KC_PROXY_PY}" --port "$2" --upstream "$1" \
    --chat-html "${KC_CHAT_HTML}" \
    --upstream-header "ngrok-skip-browser-warning=true" \
    >/tmp/kc-relay.log 2>&1 &
  KC_RELAY_PID=$!
  sleep 2
  if kill -0 "${KC_RELAY_PID}" 2>/dev/null; then
    return 0
  fi
  echo "⚠ 中継の起動に失敗しました（/tmp/kc-relay.log）" >&2
  KC_RELAY_PID=""
  return 1
}

# ---- 対話 ----
# zsh の `read -p` は coprocess 指定と解釈されて失敗する（実測）。
# プロンプトは `read "var?prompt"` の形でしか出せないのでここに集約する。
kc_ask() {  # $1=prompt → KC_ANSWER
  local answer=""
  read -r "answer?$1" || true
  KC_ANSWER="${answer}"
}

kc_pause() {  # $1=prompt（既定: Enter で閉じます）
  read -r "?${1:-Enter で閉じます...}" || true
}

# ---- ポート ----
# lsof は macOS 標準。LISTEN だけを見る（TIME_WAIT を掴まない）。
kc_port_in_use() {  # $1=port
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

kc_port_listener() {  # $1=port（表示用）
  lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null
}

kc_kill_port() {  # $1=port
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null || true
}

# ---- Ollama 同梱の llama-server / blob ----
# Ollama.app の中のバイナリを使う。PATH に無い場合の検索順を 1 箇所に閉じる。
kc_ollama_server_bin() {
  local candidates=(
    "/Applications/Ollama.app/Contents/Resources/llama-server"
    "$(command -v llama-server 2>/dev/null)"
    "/opt/homebrew/bin/llama-server"
    "/usr/local/bin/llama-server"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -n "${c}" && -x "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  return 1
}

kc_ollama_blob() {  # $1=sha256（64 hex）
  local path="${HOME}/.ollama/models/blobs/sha256-$1"
  [[ -f "${path}" ]] || return 1
  echo "${path}"
}

# ---- チャット UI 中継 ----
# auth_proxy.py を起動して UI を配る。成功時は KC_UI_PID / KC_UI_URL を設定。
# $1=upstream base url  $2=ui port  $3=停止時に SIGTERM する PID
# $4=ログ名（/tmp/<name>-ui-proxy.log）  $5=1 なら quota 有効（既定は --no-quota）
# 戻り値: 0=起動 1=起動失敗 2=必要ファイル無し 3=ポート使用中
kc_start_ui_proxy() {
  KC_UI_PID=""
  KC_UI_URL=""
  local upstream="$1" port="$2" stop_pid="$3" log_name="${4:-ui}" quota="${5:-0}"
  if [[ ! -f "${KC_PROXY_PY}" || ! -f "${KC_CHAT_HTML}" ]]; then
    return 2
  fi
  if kc_port_in_use "${port}"; then
    return 3
  fi
  local quota_args=()
  [[ "${quota}" == "1" ]] || quota_args=(--no-quota)
  python3 "${KC_PROXY_PY}" \
    --port "${port}" --host 127.0.0.1 \
    --upstream "${upstream}" \
    --chat-html "${KC_CHAT_HTML}" \
    --stop-pid "${stop_pid}" \
    "${quota_args[@]}" \
    >"/tmp/${log_name}-ui-proxy.log" 2>&1 &
  KC_UI_PID=$!
  local i
  for i in {1..40}; do
    if curl -fsS -o /dev/null -m 1 "http://127.0.0.1:${port}/" 2>/dev/null; then
      KC_UI_URL="http://127.0.0.1:${port}/"
      return 0
    fi
    kill -0 "${KC_UI_PID}" 2>/dev/null || break
    sleep 0.5
  done
  kill "${KC_UI_PID}" 2>/dev/null || true
  KC_UI_PID=""
  return 1
}
