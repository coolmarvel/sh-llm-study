#!/bin/bash
# 검증 명령 전체, 커밋 메시지 작성·산출물 전달 전에 반드시 통과 (CLAUDE.md "변경 후 자동 규칙")
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

echo "== ruff format --check"; uv run ruff format --check src tests scripts
echo "== ruff check";          uv run ruff check src tests scripts
echo "== pytest";              uv run pytest
echo "== notebooks (normalized)"; uv run python scripts/normalize_notebooks.py --check
echo "== notebooks (execute)"
mkdir -p build/notebooks
for nb in notebooks/*.ipynb; do
  echo "   $nb"
  uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
    --output-dir build/notebooks "$nb" >/dev/null 2>&1 || { echo "   FAILED: $nb"; exit 1; }
done
echo "== all passed"
