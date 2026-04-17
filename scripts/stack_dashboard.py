#!/usr/bin/env python3
"""Compact stack dashboard for PyCharm/IDE terminal usage."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CHECK_PATTERN = re.compile(r"^\[(OK|WARN|FAIL)\]\s+(.*)$")
SUMMARY_PATTERN = re.compile(r"^Summary:\s+PASS=(\d+)\s+WARN=(\d+)\s+FAIL=(\d+)$")


@dataclass(frozen=True)
class StageRule:
    name: str
    keywords: Sequence[str]
    expected: int
    hint: str


STAGE_RULES: List[StageRule] = [
    StageRule(
        name="Tmux Orchestration",
        keywords=("tmux session exists", "tmux window exists"),
        expected=7,
        hint="session + 6 key windows",
    ),
    StageRule(
        name="Infra Ports",
        keywords=("zookeeper reachable", "kafka reachable"),
        expected=2,
        hint="2181/9092 reachable",
    ),
    StageRule(
        name="Kafka Topics",
        keywords=("kafka topic exists",),
        expected=2,
        hint="traffic_raw + weather_raw",
    ),
    StageRule(
        name="Kafka Messages",
        keywords=("topic has readable message", "topic has no message"),
        expected=2,
        hint="raw topics are readable",
    ),
    StageRule(
        name="HDFS Outputs",
        keywords=("hdfs path has data files", "hdfs path has no part-*", "hdfs path not found"),
        expected=3,
        hint="3 sinks contain part-*",
    ),
]


def status_rank(status: str) -> int:
    if status == "FAIL":
        return 3
    if status == "WARN":
        return 2
    if status == "UNKNOWN":
        return 1
    return 0


def decorate(text: str, status: str, enabled: bool) -> str:
    if not enabled:
        return text

    color_map = {
        "OK": "\033[32m",
        "WARN": "\033[33m",
        "FAIL": "\033[31m",
        "UNKNOWN": "\033[36m",
    }
    reset = "\033[0m"
    return f"{color_map.get(status, '')}{text}{reset}"


def run_check(check_script: Path, strict_override: str | None) -> Dict[str, object]:
    env = os.environ.copy()
    if strict_override is not None:
        env["STRICT"] = strict_override

    proc = subprocess.run(
        ["bash", str(check_script)],
        text=True,
        capture_output=True,
        env=env,
        cwd=str(check_script.parent.parent),
        check=False,
    )

    merged_lines: List[str] = []
    for chunk in (proc.stdout, proc.stderr):
        if not chunk:
            continue
        merged_lines.extend(line for line in chunk.splitlines() if line.strip())

    checks: List[Dict[str, str]] = []
    summary: Dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for line in merged_lines:
        match = CHECK_PATTERN.match(line)
        if match:
            checks.append({"level": match.group(1), "message": match.group(2)})
            continue

        summary_match = SUMMARY_PATTERN.match(line)
        if summary_match:
            summary = {
                "PASS": int(summary_match.group(1)),
                "WARN": int(summary_match.group(2)),
                "FAIL": int(summary_match.group(3)),
            }

    if proc.returncode == 1:
        overall = "FAIL"
    elif proc.returncode == 2:
        overall = "WARN"
    elif proc.returncode == 0:
        overall = "OK"
    else:
        overall = "UNKNOWN"

    return {
        "returncode": proc.returncode,
        "overall": overall,
        "checks": checks,
        "summary": summary,
        "raw_lines": merged_lines,
    }


def evaluate_stage(rule: StageRule, checks: Sequence[Dict[str, str]]) -> Tuple[str, int, int]:
    matched = [
        item
        for item in checks
        if any(keyword in item["message"] for keyword in rule.keywords)
    ]
    if not matched:
        return "UNKNOWN", 0, rule.expected

    levels = {item["level"] for item in matched}
    if "FAIL" in levels:
        return "FAIL", len(matched), rule.expected
    if "WARN" in levels:
        return "WARN", len(matched), rule.expected
    if len(matched) < rule.expected:
        return "WARN", len(matched), rule.expected
    return "OK", len(matched), rule.expected


def render_dashboard(
    payload: Dict[str, object],
    strict_display: str,
    check_script: Path,
    interval: int,
    color: bool,
    clear_screen: bool,
    raw_tail: int,
) -> None:
    if clear_screen:
        sys.stdout.write("\033[2J\033[H")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall = str(payload["overall"])
    summary = payload["summary"]
    checks = payload["checks"]

    print("Traffic Data System - Stack Dashboard")
    print(f"Time: {now} | Overall: {decorate(overall, overall, color)} | STRICT={strict_display}")
    print(f"Check Script: {check_script}")
    print(f"Watch Interval: {interval}s")
    print("")
    print(f"Summary: PASS={summary['PASS']} WARN={summary['WARN']} FAIL={summary['FAIL']}")
    print("")
    print(f"{'Stage':<22} {'Status':<8} {'Checks':<8} Hint")
    print("-" * 78)

    worst = "OK"
    for rule in STAGE_RULES:
        stage_status, hit_count, expected = evaluate_stage(rule, checks)
        if status_rank(stage_status) > status_rank(worst):
            worst = stage_status
        status_cell = decorate(stage_status, stage_status, color)
        print(f"{rule.name:<22} {status_cell:<8} {f'{hit_count}/{expected}':<8} {rule.hint}")

    print("-" * 78)
    print(f"Pipeline Stage Worst Status: {decorate(worst, worst, color)}")
    print("")

    non_ok = [c for c in checks if c["level"] != "OK"]
    if non_ok:
        print("Attention Items:")
        for item in non_ok[:12]:
            level = item["level"]
            print(f"- {decorate(level, level, color)} {item['message']}")
        print("")

    raw_lines = payload["raw_lines"]
    if raw_tail > 0 and raw_lines:
        print(f"Latest Raw Lines (tail {raw_tail}):")
        for line in raw_lines[-raw_tail:]:
            print(f"- {line}")
        print("")

    print("Tips:")
    print("- Start (IDE mode): bash scripts/start_stack_ide.sh")
    print("- Check once:       python scripts/stack_dashboard.py")
    print("- Watch mode:       python scripts/stack_dashboard.py --watch")
    print("- Stop stack:       bash scripts/stop_stack.sh")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_check = script_dir / "check_stack.sh"

    parser = argparse.ArgumentParser(description="Compact dashboard for stack health checks.")
    parser.add_argument(
        "--check-script",
        default=str(default_check),
        help="Path to check script (default: scripts/check_stack.sh).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Refresh continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("DASHBOARD_INTERVAL", "5")),
        help="Watch refresh interval in seconds (default: 5).",
    )
    parser.add_argument(
        "--strict",
        choices=("true", "false"),
        default=None,
        help="Override STRICT for check script.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output parsed check result as JSON and exit.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear screen between refreshes in watch mode.",
    )
    parser.add_argument(
        "--raw-tail",
        type=int,
        default=8,
        help="Number of raw check lines to keep in dashboard tail (default: 8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    check_script = Path(args.check_script).resolve()
    if not check_script.exists():
        print(f"[ERROR] check script not found: {check_script}", file=sys.stderr)
        return 1

    interval = max(1, args.interval)
    color_enabled = sys.stdout.isatty() and not args.no_color
    clear_enabled = args.watch and sys.stdout.isatty() and not args.no_clear
    strict_display = args.strict if args.strict is not None else os.getenv("STRICT", "true")

    if not args.watch:
        payload = run_check(check_script, args.strict)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            render_dashboard(
                payload,
                strict_display,
                check_script,
                interval,
                color_enabled,
                clear_enabled,
                args.raw_tail,
            )
        return int(payload["returncode"])

    try:
        while True:
            payload = run_check(check_script, args.strict)
            render_dashboard(
                payload,
                strict_display,
                check_script,
                interval,
                color_enabled,
                clear_enabled,
                args.raw_tail,
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("")
        print("Dashboard stopped by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
