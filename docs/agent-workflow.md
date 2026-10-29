---
title: 에이전트 작업 계약
created: 2026-10-29
updated: 2026-10-29
domain: development
status: accepted
---

# 에이전트 작업 계약

운영 정책의 SSOT. project-seed `guides/harness-engineering.md` (2026-10-27) 를 이 프로젝트에 맞춰 채운 판이다.
이 문서만으로 실행기나 예산 차단이 생기지는 않는다. 아래 표의 "실제 적용"은 확인한 것만 적는다.

## 프로젝트 설정

| 항목 | 값 |
|---|---|
| 운영 모드 | single. 한 에이전트가 조사·구현·검증·기록. 팀 모드는 아래 "팀 모드" 조건일 때만 |
| 실행 방식 | 대화형. 무인 자동화 없음 |
| 리드 / 구현 도구 | Claude Code (모델은 세션마다 다를 수 있으니 session-log 에 필요할 때 기록) |
| 빠른 검증 | `bash scripts/verify.sh --fast` (repo 루트): ruff format --check · ruff check · pytest · 노트북 정규화 검사 |
| 통합 검증 | `bash scripts/verify.sh` (repo 루트): 빠른 검증 + `notebooks/*.ipynb` 11개 실행 -> `build/notebooks/` (약 30분) |
| 릴리스 산출물 | `uv run python scripts/build_book.py` -> `build/book/` + 바탕화면 PDF 교체 · 대시보드 `docker compose up --build -d` (8082) |
| 테스트 데이터 격리 | 단위 테스트는 작은 합성 텍스트·임시 디렉터리. `data/` (D 드라이브) 의 코퍼스·체크포인트는 읽기만, 학습 산출물은 `data/checkpoints/<run>/` 에만 쓴다 |
| 커밋·푸시 | 검증 통과 후 에이전트가 직접 (사용자 위임 2026-09-16, CLAUDE.md "커밋 컨벤션") |
| 재시도 상한 | 같은 원인으로 2회 실패하면 원인·빠진 근거를 정리해 사용자에게 보고한 뒤 진행 |
| 예산 / usage | 강제 장치 없음 (unknown). 긴 학습은 `nohup` 백그라운드, 대기는 `Monitor` 나 파일 마커 |

## 검증 계층 (어떤 변경에 무엇을 돌리나)

| 변경 | 최소 검증 | 이유 |
|---|---|---|
| 운영 문서만 (`docs/` 루트·adr·plans·guides, CLAUDE.md, AGENTS.md) | `--fast` | 노트북 출력에 영향 없음 |
| `scripts/hooks/`, 테스트, `dashboard/` | `--fast` (+ 대시보드면 컨테이너 재빌드 후 화면 확인) | 노트북이 import 하지 않음 |
| `src/shllm/`, `notebooks/`, `configs/tiny-notebook.yaml` | 전체 | 노트북 출력과 대시보드 노트북 탭 초기값(`build/notebooks/`)이 바뀐다 |
| `docs/book/` | 전체 + `build_book.py` | 교재와 노트북은 1:1, PDF 는 항상 최신 한 권 |

작업 중에는 관련 테스트 하나(`uv run pytest tests/test_x.py`)로 빠르게 돌고, 커밋 직전에 위 표의 검증을 한 번 돌린다.
돌리지 않은 검증은 session-log 에 "미실행"으로 적는다. 통과했다고 쓰지 않는다.

## 보호 장치의 실제 적용 상태

| 장치 | 상태 | 확인 방법 |
|---|---|---|
| `scripts/hooks/env-guard.sh` (.env 편집 차단) | verified (Claude Code) | `tests/test_hooks.py`: stdin JSON 합성 입력으로 차단 exit 2 / 허용 exit 0 |
| `scripts/hooks/git-add-guard.sh` (`git add -A/.`·.env staging 차단) | verified (Claude Code), 문자열 패턴이라 셸 우회는 못 막음 | 같은 테스트 |
| `scripts/hooks/format.sh` (.py 자동 포맷) | verified (Claude Code) | 같은 테스트. 최종 보증은 verify 의 `ruff format --check` |
| Remote Control 차단 (`.claude/settings.json`) | 설정만 확인 | 해당 기능 설정이며 모든 외부 전송을 막는다는 뜻은 아니다 |
| Codex·Cursor 등 다른 도구 | unknown | Claude 훅은 다른 도구에 적용되지 않는다. 그 도구의 권한 정책과 verify.sh 로 대신한다 |

2026-10-27 이전의 훅 3개는 환경변수 `CLAUDE_TOOL_INPUT` 만 읽어 아무것도 막지 못했다 (2026-10-29 재현·수정).

## 작업 흐름 (단일 모드)

1. `docs/current-state.md`, `docs/todo.md`, session-log 최신 1~2개 블록, 작업 관련 plan 과 `docs/domain-contract.md` 의 관련 규칙만 읽는다.
2. 목표·수정 범위·완료 기준을 정한다. domain-contract 에서 unknown 인 규칙은 추측으로 채우지 않고 사용자에게 묻는다.
3. 구현하고 위 "검증 계층"대로 검증한다.
4. current-state 덮어쓰기, session-log 최상단 블록, todo, (릴리스급이면) changelog, 버전을 한 번에 갱신한다.

## 팀 모드 (선택)

서로 독립인 작업이 둘 이상이고 병렬 이득이 조율 비용보다 클 때만 쓴다 (예: 장 두 개의 교재 보강을 동시에, 구현과 독립 리뷰).
위임할 때는 목표·읽을 파일·수정 허용 경로·완료 기준을 넘기고, worker 는 변경 파일과 실행한 검사 명령·결과를 돌려준다.
통합 문서(current-state·session-log·todo·changelog·버전)는 리드만 고친다. 같은 파일을 두 worker 가 동시에 고치지 않는다.
리뷰어는 구현자의 결론이 아니라 diff 와 완료 기준에서 출발한다. 모델끼리 동의한 것은 정답의 근거가 아니다.

## 결과 보고

완료 보고에는 변경 파일, 실행한 검사(명령·결과), 미실행 검사, 남은 위험을 적는다.
timeout·환경 오류·검사 미실행은 완료가 아니다. 테스트를 통과시키려고 기대값을 바꿨다면 그 근거를 domain-contract 나 session-log 에 남긴다.
