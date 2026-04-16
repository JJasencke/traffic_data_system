#!/usr/bin/env bash
set -Eeuo pipefail

SESSION_NAME="${SESSION_NAME:-traffic-stack}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

KAFKA_HOME="${KAFKA_HOME:-/mnt/d/bigdata/apps/kafka}"
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-localhost:9092}"
KAFKA_TRAFFIC_TOPIC="${KAFKA_TRAFFIC_TOPIC:-traffic_raw}"
KAFKA_WEATHER_TOPIC="${KAFKA_WEATHER_TOPIC:-weather_raw}"

TRAFFIC_DETAIL_OUTPUT_PATH="${TRAFFIC_DETAIL_OUTPUT_PATH:-hdfs://localhost:9000/traffic/history/traffic_detail}"
WEATHER_OUTPUT_PATH="${WEATHER_OUTPUT_PATH:-hdfs://localhost:9000/traffic/history/weather}"
AVG_SPEED_OUTPUT_PATH="${AVG_SPEED_OUTPUT_PATH:-hdfs://localhost:9000/traffic/history/avg_speed}"

TOPIC_TIMEOUT_MS="${TOPIC_TIMEOUT_MS:-15000}"
PORT_WAIT_SECONDS="${PORT_WAIT_SECONDS:-20}"
STRICT="${STRICT:-true}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

ok() {
  echo "[OK] $1"
  PASS_COUNT=$((PASS_COUNT + 1))
}

warn() {
  echo "[WARN] $1"
  WARN_COUNT=$((WARN_COUNT + 1))
}

fail() {
  echo "[FAIL] $1"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

require_cmd() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    ok "command exists: ${cmd}"
  else
    fail "missing command: ${cmd}"
  fi
}

maybe_fail_or_warn() {
  local msg="$1"
  if [[ "${STRICT,,}" == "true" ]]; then
    fail "${msg}"
  else
    warn "${msg}"
  fi
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"

  for ((i=1; i<=PORT_WAIT_SECONDS; i++)); do
    if bash -c "exec 3<>/dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      ok "${name} reachable on ${host}:${port}"
      return 0
    fi
    sleep 1
  done

  fail "${name} unreachable on ${host}:${port} after ${PORT_WAIT_SECONDS}s"
  return 1
}

check_tmux_windows() {
  local expected=(zookeeper kafka collector traffic_detail weather_detail avg_speed)

  if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    fail "tmux session not found: ${SESSION_NAME}"
    return
  fi

  ok "tmux session exists: ${SESSION_NAME}"

  local windows
  windows="$(tmux list-windows -t "${SESSION_NAME}" -F '#W' 2>/dev/null || true)"

  for name in "${expected[@]}"; do
    if grep -Fxq "${name}" <<<"${windows}"; then
      ok "tmux window exists: ${name}"
    else
      fail "tmux window missing: ${name}"
    fi
  done
}

check_kafka_topic_exists() {
  local topic="$1"
  local list_output

  if ! list_output="$("${KAFKA_HOME}/bin/kafka-topics.sh" --bootstrap-server "${KAFKA_BOOTSTRAP_SERVERS}" --list 2>/dev/null)"; then
    fail "unable to list kafka topics via ${KAFKA_BOOTSTRAP_SERVERS}"
    return
  fi

  if grep -Fxq "${topic}" <<<"${list_output}"; then
    ok "kafka topic exists: ${topic}"
  else
    fail "kafka topic missing: ${topic}"
  fi
}

check_topic_has_message() {
  local topic="$1"
  local tmp
  tmp="$(mktemp)"

  "${KAFKA_HOME}/bin/kafka-console-consumer.sh" \
    --bootstrap-server "${KAFKA_BOOTSTRAP_SERVERS}" \
    --topic "${topic}" \
    --max-messages 1 \
    --timeout-ms "${TOPIC_TIMEOUT_MS}" \
    >"${tmp}" 2>/dev/null || true

  if [[ -s "${tmp}" ]]; then
    ok "topic has readable message: ${topic}"
  else
    maybe_fail_or_warn "topic has no message within ${TOPIC_TIMEOUT_MS}ms: ${topic}"
  fi

  rm -f "${tmp}"
}

check_hdfs_path_has_data() {
  local path="$1"
  local label="$2"

  if ! hdfs dfs -test -e "${path}"; then
    maybe_fail_or_warn "hdfs path not found: ${label} (${path})"
    return
  fi

  local part_count
  part_count="$(hdfs dfs -find "${path}" -name 'part-*' 2>/dev/null | wc -l | tr -d ' ')"

  if [[ "${part_count}" =~ ^[0-9]+$ ]] && (( part_count > 0 )); then
    ok "hdfs path has data files: ${label} (${part_count})"
  else
    maybe_fail_or_warn "hdfs path has no part-* data files yet: ${label} (${path})"
  fi
}

echo "========== Traffic Data System Smoke Check =========="
echo "SESSION_NAME=${SESSION_NAME}"
echo "KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS}"
echo "STRICT=${STRICT}"
echo "====================================================="

require_cmd bash
require_cmd tmux
require_cmd hdfs
require_cmd grep
require_cmd wc

if [[ ! -d "${KAFKA_HOME}" ]]; then
  fail "KAFKA_HOME not found: ${KAFKA_HOME}"
else
  ok "KAFKA_HOME found: ${KAFKA_HOME}"
fi

if [[ ! -x "${KAFKA_HOME}/bin/kafka-topics.sh" ]]; then
  fail "kafka-topics.sh not executable: ${KAFKA_HOME}/bin/kafka-topics.sh"
fi

if [[ ! -x "${KAFKA_HOME}/bin/kafka-console-consumer.sh" ]]; then
  fail "kafka-console-consumer.sh not executable: ${KAFKA_HOME}/bin/kafka-console-consumer.sh"
fi

check_tmux_windows
wait_for_port "127.0.0.1" 2181 "zookeeper"
wait_for_port "127.0.0.1" 9092 "kafka"

check_kafka_topic_exists "${KAFKA_TRAFFIC_TOPIC}"
check_kafka_topic_exists "${KAFKA_WEATHER_TOPIC}"

check_topic_has_message "${KAFKA_TRAFFIC_TOPIC}"
check_topic_has_message "${KAFKA_WEATHER_TOPIC}"

check_hdfs_path_has_data "${TRAFFIC_DETAIL_OUTPUT_PATH}" "traffic_detail"
check_hdfs_path_has_data "${WEATHER_OUTPUT_PATH}" "weather_detail"
check_hdfs_path_has_data "${AVG_SPEED_OUTPUT_PATH}" "avg_speed"

echo "====================================================="
echo "Summary: PASS=${PASS_COUNT} WARN=${WARN_COUNT} FAIL=${FAIL_COUNT}"

if (( FAIL_COUNT > 0 )); then
  exit 1
fi

if (( WARN_COUNT > 0 )); then
  exit 2
fi

exit 0
