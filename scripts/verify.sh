#!/bin/bash
# 검증 명령, 커밋 메시지 작성·산출물 전달 전에 반드시 통과 (CLAUDE.md "변경 후 자동 규칙")
#   bash scripts/verify.sh          전체: 빠른 검사 + 노트북 11개 실행 (약 30분, build/notebooks/ 갱신)
#   bash scripts/verify.sh --fast   빠른 검사만: ruff · pytest · 노트북 정규화 (1분 안팎)
# 어느 쪽을 써야 하는지는 docs/agent-workflow.md "검증 계층".
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

FAST=0
[[ "${1:-}" == "--fast" ]] && FAST=1

echo "== ruff format --check"; uv run ruff format --check src tests scripts
echo "== ruff check";          uv run ruff check src tests scripts
echo "== pytest";              uv run pytest
echo "== notebooks (normalized)"; uv run python scripts/normalize_notebooks.py --check
if [[ $FAST -eq 1 ]]; then
  echo "== fast checks passed (notebooks NOT executed)"
  exit 0
fi
echo "== notebooks (execute)"
mkdir -p build/notebooks
for nb in notebooks/*.ipynb; do
  echo "   $nb"
  uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
    --output-dir build/notebooks "$nb" >/dev/null 2>&1 || { echo "   FAILED: $nb"; exit 1; }
done
echo "== all passed"
