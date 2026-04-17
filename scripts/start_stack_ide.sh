#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-traffic-stack}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SESSION_NAME="${SESSION_NAME:-traffic-stack}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[INFO] tmux session already running, skip start."
else
  AUTO_ATTACH=false bash "${SCRIPT_DIR}/start_stack.sh"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/stack_dashboard.py" --watch --interval "${DASHBOARD_INTERVAL:-5}"
