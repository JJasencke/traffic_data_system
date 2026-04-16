#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="${SESSION_NAME:-traffic-stack}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KAFKA_HOME="${KAFKA_HOME:-/mnt/d/bigdata/apps/kafka}"
SPARK_HOME="${SPARK_HOME:-/mnt/d/bigdata/apps/spark}"
SPARK_KAFKA_PACKAGE="${SPARK_KAFKA_PACKAGE:-org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
ENABLE_MONITOR="${ENABLE_MONITOR:-true}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd}" >&2
    exit 1
  fi
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout="$4"

  for ((i=1; i<=timeout; i++)); do
    if bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      echo "[OK] ${name} is ready on ${host}:${port}"
      return 0
    fi
    sleep 1
  done

  echo "[ERROR] ${name} not ready on ${host}:${port} after ${timeout}s" >&2
  return 1
}

new_window() {
  local name="$1"
  local command="$2"
  tmux new-window -t "${SESSION_NAME}" -n "${name}"
  tmux send-keys -t "${SESSION_NAME}:${name}" "${command}" C-m
}

require_cmd tmux
require_cmd bash

[[ -d "${KAFKA_HOME}" ]] || { echo "[ERROR] KAFKA_HOME not found: ${KAFKA_HOME}" >&2; exit 1; }
[[ -d "${SPARK_HOME}" ]] || { echo "[ERROR] SPARK_HOME not found: ${SPARK_HOME}" >&2; exit 1; }
[[ -f "${REPO_ROOT}/src/ingestion/jobs/run_collector_job.py" ]] || { echo "[ERROR] repo path invalid: ${REPO_ROOT}" >&2; exit 1; }

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "[ERROR] tmux session already exists: ${SESSION_NAME}" >&2
  echo "Run: tmux attach -t ${SESSION_NAME} OR scripts/stop_stack.sh" >&2
  exit 1
fi

echo "[INFO] Starting tmux session: ${SESSION_NAME}"
tmux new-session -d -s "${SESSION_NAME}" -n "zookeeper"
tmux send-keys -t "${SESSION_NAME}:zookeeper" "cd \"${KAFKA_HOME}\" && exec bin/zookeeper-server-start.sh config/zookeeper.properties" C-m

new_window "kafka" "cd \"${KAFKA_HOME}\" && exec bin/kafka-server-start.sh config/server.properties"

echo "[INFO] Waiting for infra services..."
wait_for_port "127.0.0.1" 2181 "zookeeper" "${WAIT_SECONDS}"
wait_for_port "127.0.0.1" 9092 "kafka" "${WAIT_SECONDS}"

echo "[INFO] Starting 5 runtime windows..."
new_window "collector" "cd \"${REPO_ROOT}\" && export PYTHONPATH=\"${REPO_ROOT}/src\" && exec \"${PYTHON_BIN}\" src/ingestion/jobs/run_collector_job.py"
new_window "traffic_detail" "cd \"${SPARK_HOME}\" && export PYTHONPATH=\"${REPO_ROOT}/src\" && exec bin/spark-submit --packages \"${SPARK_KAFKA_PACKAGE}\" \"${REPO_ROOT}/src/streaming/jobs/traffic_detail_stream_job.py\""
new_window "weather_detail" "cd \"${SPARK_HOME}\" && export PYTHONPATH=\"${REPO_ROOT}/src\" && exec bin/spark-submit --packages \"${SPARK_KAFKA_PACKAGE}\" \"${REPO_ROOT}/src/streaming/jobs/weather_detail_stream_job.py\""
new_window "avg_speed" "cd \"${SPARK_HOME}\" && export PYTHONPATH=\"${REPO_ROOT}/src\" && exec bin/spark-submit --packages \"${SPARK_KAFKA_PACKAGE}\" \"${REPO_ROOT}/src/streaming/jobs/traffic_avg_speed_stream_job.py\""
new_window "traffic_watch" "cd \"${KAFKA_HOME}\" && exec bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic \"${KAFKA_TRAFFIC_TOPIC:-traffic_raw}\""

if [[ "${ENABLE_MONITOR,,}" != "true" ]]; then
  tmux kill-window -t "${SESSION_NAME}:traffic_watch"
fi

tmux select-window -t "${SESSION_NAME}:collector" 2>/dev/null || true

echo "[INFO] Session ready: ${SESSION_NAME}"
echo "[INFO] Attach with: tmux attach -t ${SESSION_NAME}"
echo "[INFO] Stop with: ${REPO_ROOT}/scripts/stop_stack.sh"

exec tmux attach -t "${SESSION_NAME}"
