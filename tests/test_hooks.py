"""하네스 훅 회귀 검사 (ADR-0001).

Claude Code 는 훅에 `{"tool_name": ..., "tool_input": {...}}` JSON 을 stdin 으로 넘긴다.
2026-10-27 project-seed 감사에서 환경변수만 읽던 훅이 아무것도 막지 못한 것이 재현됐다.
그래서 "파일이 있다"가 아니라 "합성 입력을 넣으면 실제로 막는다/통과시킨다"를 검사한다.
차단은 exit 2, 통과는 exit 0.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "scripts" / "hooks"


def run_hook(name: str, tool_name: str, tool_input: dict) -> int:
    """훅 하나를 실제 Claude Code 와 같은 방식(stdin JSON)으로 실행하고 종료 코드를 돌려준다."""
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_TOOL_INPUT"}
    payload = json.dumps({"tool_name": tool_name, "tool_input": tool_input})
    result = subprocess.run(
        ["bash", str(HOOKS / name)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return result.returncode


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/repo/.env", 2),
        (".env", 2),
        ("/repo/.env.local", 2),
        ("/repo/.env.example", 0),
        ("/repo/src/shllm/config.py", 0),
        ("/repo/docs/environment.md", 0),
    ],
)
def test_env_guard(path, expected):
    assert run_hook("env-guard.sh", "Write", {"file_path": path}) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git add -A", 2),
        ("git add --all", 2),
        ("git add .", 2),
        ("git add . && git commit -m x", 2),
        ("git add .env", 2),
        ("git add docs/.env.local", 2),
        ("git add .env.example .env", 2),
        ("git add .env.example", 0),
        ("git add src/shllm/model.py tests/test_model.py", 0),
        ("git status", 0),
        # 2026-10-29 오탐: 문서 본문에 적힌 문구가 명령으로 잡혔다
        ("echo '| `git add -A/.`·.env staging 차단 |' >> CLAUDE.md", 0),
        ("python3 - <<'EOF'\ns = '`git add . && ...` 누락'\nEOF", 0),
        ("cd /repo && git add -A", 2),
        ("git status; git add .env.local", 2),
        ("git add -- .env", 2),
    ],
)
def test_git_add_guard(command, expected):
    assert run_hook("git-add-guard.sh", "Bash", {"command": command}) == expected


def test_guards_pass_on_malformed_input():
    """JSON 이 아닌 입력은 막지 않고 통과시킨다 (훅 오류로 작업 전체가 멈추지 않게)."""
    result = subprocess.run(
        ["bash", str(HOOKS / "env-guard.sh")], input="not json", capture_output=True, text=True
    )
    assert result.returncode == 0


def test_format_hook_formats_python(tmp_path):
    """PostToolUse format 훅이 stdin 의 file_path 를 받아 실제로 ruff format 을 적용한다."""
    target = tmp_path / "messy.py"
    target.write_text("x=[1,2 ,3]\n", encoding="utf-8")
    assert run_hook("format.sh", "Write", {"file_path": str(target)}) == 0
    assert target.read_text(encoding="utf-8") == "x = [1, 2, 3]\n"
