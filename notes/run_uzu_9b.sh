#!/bin/zsh
# Run uzu 9B-M locally on Apple M3 with the M3 speculator shape enabled.
# Usage:
#   notes/run_uzu_9b.sh "プロンプト"
#   notes/run_uzu_9b.sh            # interactive chat
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ROOT="${HOME}/Myproject/uzu"
MODEL="${HOME}/.cache/uzu-stage"
SHAPES="${MODEL}/speculator/shapes.json"
ENCODING="${MODEL}/encoding.json"
CLI="${ROOT}/target/release/cli"

BACKUP="${SCRIPT_DIR}/uzu-backup"
BACKUP_SHAPES="${BACKUP}/shapes.json"
BACKUP_ENCODING="${BACKUP}/encoding.json"
BACKUP_CLI="${BACKUP}/cli"
BACKUP_CONFIG="${BACKUP}/config.json"
BACKUP_TOKENIZER="${BACKUP}/tokenizer.json"
BACKUP_SPEC_CONFIG="${BACKUP}/speculator-config.json"

restore_file() {
  local source="$1"
  local destination="$2"
  if [[ ! -f "${destination}" && -f "${source}" ]]; then
    mkdir -p "$(dirname "${destination}")"
    cp "${source}" "${destination}"
    echo "restored from backup: ${destination}" >&2
  fi
}

# Free unified memory. Ollama resident models cause swap thrashing on 16GB M3.
# Set UZU_KEEP_OLLAMA=1 to skip this.
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

# Restore the locally patched CLI if a rebuild/clean removed it.
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

# Restore small configs if a re-download or cleanup removed them.
restore_file "${BACKUP_CONFIG}" "${MODEL}/config.json"
restore_file "${BACKUP_TOKENIZER}" "${MODEL}/tokenizer.json"
restore_file "${BACKUP_SPEC_CONFIG}" "${MODEL}/speculator/config.json"

# Model weights and configs are required. They are large, so do not auto-download.
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

# The upstream shapes.json only ships M5/M6. Restore from backup if missing or
# corrupt, then add M3 idempotently because a model re-download overwrites it.
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

# Qwen3.5 uses the hanashi qwen3.5 encoding. The staging download does not
# include encoding.json. Keep the pinned file; repair from backup only when
# missing, invalid, or different.
if [[ ! -f "${ENCODING}" ]]; then
  if [[ -f "${BACKUP_ENCODING}" ]]; then
    cp "${BACKUP_ENCODING}" "${ENCODING}"
    echo "restored encoding.json from backup: ${ENCODING}" >&2
  fi
fi

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

cd "${ROOT}"
if [[ $# -eq 0 ]]; then
  exec "${CLI}" --model "${MODEL}" --reasoning-effort disabled
else
  exec "${CLI}" --model "${MODEL}" --reasoning-effort disabled -m "$*"
fi
