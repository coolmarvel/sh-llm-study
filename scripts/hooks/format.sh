#!/bin/bash
# PostToolUse hook: 편집된 .py 파일을 ruff 로 포맷 (ADR-0001 참고)
set -uo pipefail

FILE_PATH=$(python3 -c "
import json, os
data = json.loads(os.environ.get('CLAUDE_TOOL_INPUT', '{}'))
print(data.get('file_path', ''))
" 2>/dev/null) || exit 0

[[ -z "$FILE_PATH" ]] && exit 0
[[ "$FILE_PATH" != *.py ]] && exit 0
[[ -f "$FILE_PATH" ]] || exit 0

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.." || exit 0
uv run --quiet ruff format "$FILE_PATH" >/dev/null 2>&1
uv run --quiet ruff check --fix --quiet "$FILE_PATH" >/dev/null 2>&1
exit 0
