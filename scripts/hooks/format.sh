#!/bin/bash
# PostToolUse hook: 편집된 .py 파일을 ruff 로 포맷 (ADR-0001 참고)
# 입력은 stdin 의 JSON (tool_input.file_path). 회귀 검사: tests/test_hooks.py
set -uo pipefail

FILE_PATH=$(python3 "$(dirname "$0")/read-tool-input.py" file_path) || exit 0

[[ -z "$FILE_PATH" ]] && exit 0
[[ "$FILE_PATH" != *.py ]] && exit 0
[[ -f "$FILE_PATH" ]] || exit 0

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.." || exit 0
uv run --quiet ruff format "$FILE_PATH" >/dev/null 2>&1
uv run --quiet ruff check --fix --quiet "$FILE_PATH" >/dev/null 2>&1
exit 0
