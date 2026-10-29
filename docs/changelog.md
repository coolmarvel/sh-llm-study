---
title: 체인지로그
created: 2026-09-16
updated: 2026-10-29
domain: development
---

# 체인지로그 (릴리스 단위)

사람용 요약, 릴리스(버전 승격·배포) 단위로만 기록한다. 세부 이력은 `session-log.md`.

## v0.1.6, 2026-10-29
- 하네스 훅 3개(.env 편집 차단, git add 가드, 자동 포맷)가 실제로는 동작하지 않던 결함 수정 + 회귀 테스트
- `verify.sh --fast` 추가, 변경 종류별 검증 계층
- 운영 문서 추가: current-state(인수인계), agent-workflow(작업 계약), domain-contract(판정 규칙·수용 예시)

<!-- ## vX.Y.Z. YYYY-MM-DD
- 변경 요약 -->
