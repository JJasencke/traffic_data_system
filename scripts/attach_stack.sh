#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="${SESSION_NAME:-traffic-stack}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] Missing command: tmux" >&2
  exit 1
fi

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[INFO] Session not found: ${SESSION_NAME}"
  exit 0
fi

exec tmux attach -t "${SESSION_NAME}"
