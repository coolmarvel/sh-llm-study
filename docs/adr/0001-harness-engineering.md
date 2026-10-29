---
title: ADR-0001 하네스 엔지니어링 도입
created: 2026-09-16
updated: 2026-10-29
status: accepted
---

# ADR-0001: 하네스 엔지니어링 도입

## 상태

Accepted (project-seed 보일러플레이트 기본 탑재). 2026-10-29 project-seed 감사(2026-10-27) 반영으로 개정.
운영 원칙의 채택이며, 개별 보호 장치가 작동한다는 보증은 아니다. 실제 적용 상태는 `../agent-workflow.md`.

## 맥락

AI 에이전트와의 협업에서 지시문(프롬프트)만으로는 규칙이 지켜지지 않는다. 세션이 바뀌면 잊히고,
컨텍스트가 길어지면 무시된다. 선행 프로젝트들에서 반복 확인된 문제:
`.env` 를 직접 고치거나, `git add -A` 로 민감 파일이 staging 되거나, 검증 없이 산출물이 나가는 사고.

## 결정

규칙을 **실행 가능한 제약과 검사로 연결**한다 (Constrain, Verify, Correct):

- **Hooks** (`scripts/hooks/` + `.claude/settings.json`): PreToolUse 로 위험 행동을 차단(exit 2)하고,
  PostToolUse 로 포맷을 자동화한다. `env-guard`/`git-add-guard`/`format`.
  시드 판은 환경변수 `CLAUDE_TOOL_INPUT` 만 읽어 공식 stdin 입력을 받지 못했고, 그래서 아무것도 막지 못했다
  (2026-10-29 이 저장소에서 재현). stdin JSON 을 읽게 고치고 `tests/test_hooks.py` 로 차단/허용을 회귀 검사한다.
  Claude Code 전용이며 Codex 등 다른 도구에는 적용되지 않는다.
- **Settings** (`.claude/settings.json`): 에이전트 툴의 외부 동기화를 프로젝트 단위로 끈다. Claude Code Remote Control
  (`disableRemoteControl: true`, `remoteControlAtStartup: false`, `autoUploadSessions: false`). 해당 기능의 설정이며 모든 외부 전송을 막는 보장은 아니다.
  사용자 설정(`~/.claude/settings.json`)에도 같은 값을 두되, 새 PC 에서 잊어도 프로젝트 파일이 막도록 양쪽에 둔다.
- **Commands** (`.claude/commands/`): 리뷰 체크리스트를 슬래시 커맨드로, `/review-security`, `/deploy-check`.
  결과는 Critical/Warning/Info 로 분류하고 파일:라인을 인용하게 한다.
- **ADR** (`docs/adr/`): 구조적 결정은 근거와 함께 박제해 다음 세션/에이전트가 재논쟁하지 않게 한다.
- **문서 프로세스**: current-state 는 현재 인수인계, session-log 는 과거 이력. 최신 1~2개 블록부터 읽는다 (CLAUDE.md 부팅 프로토콜).
- **도메인 판정**: `docs/domain-contract.md` 에 판정 규칙·수용 예시·unknown 을 기록하고 예시를 테스트와 연결한다.
- **검증 계층**: `verify.sh --fast`(ruff·pytest·노트북 정규화) / 전체(+노트북 실행). 변경 위험에 맞춰 검증 비용을 나눈다.
- **팀 운영**: 기본은 단일 에이전트. 위임할 때의 범위·증거·통합 규칙은 `docs/agent-workflow.md`.

## 근거

- 지시문은 어겨져도 흔적이 없지만, 훅은 차단 로그가 남고 우회가 명시적이다.
- 단, 훅 파일이 있다는 것과 실제로 막는다는 것은 다르다 (위 stdin 결함). 강제 여부는 재현 테스트로 확인한다.
- 대안(전부 프롬프트에 의존)은 선행 프로젝트에서 이미 실패 사례가 축적됨.

## 결과

- (+) 근거 추적·재현·인수인계가 쉬워진다. Claude Code 세션에서는 최소 안전선이 테스트로 보증된다.
- (-) 훅·검사 유지보수 비용. 정당한 예외는 사용자가 직접 수행해야 한다.
- (-) 다른 도구에는 훅이 없다. 그때 남는 공통 안전선은 verify.sh 와 이 문서의 규칙뿐이다.
- (-) 판정 규칙이 틀리면 테스트도 틀린 결과를 통과시킨다. 기대값은 사용자 확인이 필요하다 (domain-contract).
