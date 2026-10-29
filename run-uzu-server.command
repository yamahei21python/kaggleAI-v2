#!/bin/zsh
# uzu 9B-M を OpenAI 互換サーバとして起動する。
# opencode からも llama-chat.html からも使える。
#
# 使い方:
#   ./run-uzu-server.command                 # ダブルクリック or 実行
#   UZU_REASONING_EFFORT=default ./run-uzu-server.command   # thinking ON
#   UZU_PORT=9940 ./run-uzu-server.command
#
# 環境変数:
#   UZU_PORT              待受ポート (既定 9931)
#   UZU_REASONING_EFFORT  リクエストが指定しない場合の既定 (disabled|default|low|medium|high|xhigh, 既定 disabled)
#   UZU_KEEP_OLLAMA=1     Ollama 常駐モデルを止めない
#   UZU_NO_OPEN=1         ブラウザを自動で開かない
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT="${HOME}/Myproject/uzu"
MODEL="${HOME}/.cache/uzu-stage"
SHAPES="${MODEL}/speculator/shapes.json"
ENCODING="${MODEL}/encoding.json"
CLI="${ROOT}/target/release/cli"

BACKUP="${SCRIPT_DIR}/notes/uzu-backup"
BACKUP_SHAPES="${BACKUP}/shapes.json"
BACKUP_CLI="${BACKUP}/cli"
BACKUP_CONFIG="${BACKUP}/config.json"
BACKUP_TOKENIZER="${BACKUP}/tokenizer.json"
BACKUP_SPEC_CONFIG="${BACKUP}/speculator-config.json"

HOST="127.0.0.1"
PORT="${UZU_PORT:-9931}"
REASONING="${UZU_REASONING_EFFORT:-default}"

restore_file() {
  local source="$1"
  local destination="$2"
  if [[ ! -f "${destination}" && -f "${source}" ]]; then
    mkdir -p "$(dirname "${destination}")"
    cp "${source}" "${destination}"
    echo "restored from backup: ${destination}" >&2
  fi
}

# 16GB M3 では Ollama 常駐がスワップを起こす。
if [[ "${UZU_KEEP_OLLAMA:-0}" != "1" ]] && command -v ollama >/dev/null 2>&1; then
  loaded="$(ollama ps 2>/dev/null | awk 'NR > 1 && NF > 0 {print $1}' || true)"
  if [[ -n "${loaded}" ]]; then
    echo "Stopping Ollama models to free unified memory:" >&2
    while IFS= read -r model; do
      [[ -n "${model}" ]] || continue
      echo "  ${model}" >&2
      ollama stop "${model}" >/dev/null 2>&1 || true
    done <<< "${loaded}"
  fi
fi

# ローカル修正版 CLI が消えていたら backup から戻す。
if [[ ! -x "${CLI}" ]]; then
  if [[ -x "${BACKUP_CLI}" ]]; then
    mkdir -p "$(dirname "${CLI}")"
    cp "${BACKUP_CLI}" "${CLI}"
    chmod +x "${CLI}"
    echo "restored CLI from backup: ${CLI}" >&2
  else
    echo "missing or not executable: ${CLI}" >&2
    exit 1
  fi
fi

restore_file "${BACKUP_CONFIG}" "${MODEL}/config.json"
restore_file "${BACKUP_TOKENIZER}" "${MODEL}/tokenizer.json"
restore_file "${BACKUP_SPEC_CONFIG}" "${MODEL}/speculator/config.json"

for file in \
  "${MODEL}/model.safetensors" \
  "${MODEL}/speculator/model.safetensors" \
  "${MODEL}/config.json" \
  "${MODEL}/tokenizer.json" \
  "${MODEL}/speculator/config.json"; do
  if [[ ! -f "${file}" ]]; then
    echo "missing: ${file}" >&2
    exit 1
  fi
done

# 上流 shapes.json は M5/M6 のみ。M3 エントリを冪等に保証する。
if [[ ! -f "${SHAPES}" ]]; then
  if [[ -f "${BACKUP_SHAPES}" ]]; then
    mkdir -p "$(dirname "${SHAPES}")"
    cp "${BACKUP_SHAPES}" "${SHAPES}"
    echo "restored shapes.json from backup: ${SHAPES}" >&2
  else
    echo "missing: ${SHAPES}" >&2
    exit 1
  fi
fi

python3 - "${SHAPES}" "${BACKUP_SHAPES}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
backup = Path(sys.argv[2])

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    data = None

needs_write = False
if not isinstance(data, dict):
    if backup.is_file():
        data = json.loads(backup.read_text(encoding="utf-8"))
        needs_write = True
        print(f"restored invalid shapes.json from backup: {path}")
    else:
        raise ValueError(f"expected JSON object: {path}")

if "Apple M3" not in data:
    data["Apple M3"] = {
        "tree_budget": 16,
        "max_tree_depth": 16,
        "dflash_depth_override": None,
        "construction_method": {
            "type": "Weaver",
            "rounds": 16,
            "expand_per_round": 4,
            "expand_width": 4,
        },
    }
    needs_write = True

if needs_write:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"patched Apple M3: {path}")
PY

# Qwen3.5 は hanashi qwen3.5 encoding。無い/違う時だけ固定値へ戻す。
python3 - "${ENCODING}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = [{"type": "hanashi", "name": "qwen3.5"}]

try:
    actual = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    actual = None

if actual != expected:
    path.write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"pinned encoding: {path}")
PY

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port ${PORT} already in use. stop the process or set UZU_PORT." >&2
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN | sed -n '2,3p' >&2
  exit 1
fi

UI_PORT="${UZU_UI_PORT:-8899}"
UI_PID=""
SERVER_PID=""
CLEANED=0

validate_port() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    echo "invalid ${name}: ${value}" >&2
    exit 1
  fi
}

validate_port UZU_PORT "${PORT}"
validate_port UZU_UI_PORT "${UI_PORT}"

if [[ "${PORT}" == "${UI_PORT}" ]]; then
  echo "UZU_PORT and UZU_UI_PORT must differ" >&2
  exit 1
fi

cleanup() {
  (( CLEANED )) && return 0
  CLEANED=1
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${UI_PID}" ]]; then
    kill "${UI_PID}" 2>/dev/null || true
    wait "${UI_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# file:// から 127.0.0.1 への fetch は Chrome の Private Network Access で
# 遮断される。llama-chat.html は http で配信する。
if lsof -nP -iTCP:"${UI_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "UI port ${UI_PORT} in use; set UZU_UI_PORT to another port." >&2
  exit 1
fi

# UI は静的サーバーではなく auth_proxy.py で配信する。
# 理由: /stop（停止ボタン）を受けるには動的なエンドポイントが要る。
# プロキシは UI を / で配り、API は上流（uzu）へ中継する。
# これで Kaggle 側と同じ HTML・同じ操作感になる。
UI_URL="http://127.0.0.1:${UI_PORT}/"

PROXY_PY="${SCRIPT_DIR}/kaggle/step2/auth_proxy.py"
if [[ ! -f "${PROXY_PY}" ]]; then
  echo "auth_proxy.py が見つかりません: ${PROXY_PY}" >&2
  echo "UI 無しで続行します。" >&2
  UI_URL="(unavailable)"
else
  # --stop-pid に自分（ランチャー）の PID を渡す。
  # ブラウザの停止ボタン → proxy の /stop → ランチャーに SIGTERM
  # → trap cleanup が uzu と UI をまとめて落とす。
  # --no-quota: uzu に GPU クォータは無いので残り時間は出さない。
  python3 "${PROXY_PY}" \
    --port "${UI_PORT}" \
    --host 127.0.0.1 \
    --upstream "http://${HOST}:${PORT}" \
    --chat-html "${SCRIPT_DIR}/llama-chat.html" \
    --stop-pid $$ \
    --no-quota \
    >/tmp/uzu-ui-proxy.log 2>&1 &
  UI_PID=$!
fi

ui_ready=0
for _ in {1..40}; do
  if curl -fsS -o /dev/null -m 1 2>/dev/null "http://127.0.0.1:${UI_PORT}/"; then
    ui_ready=1
    break
  fi
  if ! kill -0 "${UI_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if (( ui_ready == 0 )); then
  echo "UI server failed to start on ${UI_PORT}; continuing without browser UI." >&2
  if [[ -n "${UI_PID}" ]]; then
    kill "${UI_PID}" 2>/dev/null || true
    wait "${UI_PID}" 2>/dev/null || true
  fi
  UI_PID=""
  UI_URL="(unavailable)"
fi

echo "============================================================"
echo "  uzu 9B-M server"
echo "  model     : ${MODEL}"
echo "  endpoint  : http://${HOST}:${PORT}"
echo "  reasoning : ${REASONING} (UZU_REASONING_EFFORT)"
echo "  opencode  : opencode --model uzu-local/uzu-stage"
echo "  browser   : ${UI_URL}"
echo "  stop      : Ctrl+C"
echo "============================================================"
echo

if [[ "${UZU_NO_OPEN:-0}" != "1" ]]; then
  : # ブラウザはモデル準備完了後に開く
fi

cd "${ROOT}"
UZU_REASONING_EFFORT="${REASONING}" "${CLI}" server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" &
SERVER_PID=$!

# モデルロード完了（/v1/models が応答）まで待ってからブラウザを開く。
server_ready=0
for _ in {1..600}; do
  if curl -fsS -o /dev/null -m 1 2>/dev/null "http://${HOST}:${PORT}/v1/models"; then
    server_ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if (( server_ready == 0 )); then
  echo "server did not become ready; see the log above." >&2
fi

if (( server_ready == 1 )) && [[ "${UZU_NO_OPEN:-0}" != "1" ]] && [[ "${UI_URL}" != "(unavailable)" ]]; then
  echo "model ready: opening ${UI_URL}"
  open "${UI_URL}" >/dev/null 2>&1 || true
fi

server_status=0
wait "${SERVER_PID}" || server_status=$?
exit "${server_status}"