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

for win in traffic_watch avg_speed weather_detail traffic_detail collector kafka zookeeper; do
  tmux send-keys -t "${SESSION_NAME}:${win}" C-c 2>/dev/null || true
done

sleep 2
tmux kill-session -t "${SESSION_NAME}"

echo "[OK] Session stopped: ${SESSION_NAME}"
