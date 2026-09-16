---
title: 세션 로그
created: 2026-09-16
updated: 2026-09-16
domain: development
---

# 세션 로그 (최신이 위)

이 파일이 **"언제 무슨 일이 있었나"의 SSOT**다. 세션마다 최상단에 블록 추가.
(커밋/푸시는 사용자가 직접·성긴 단위 — git history 를 이력 SSOT 로 삼지 않는다.)

블록 형식: `## YYYY-MM-DD — 제목` 아래에 **요청/피드백 → 수정 → 검증 → 다음** 순서로 간결하게.

## 2026-09-16 — 킥오프 완료

- **요청**: "나만의 LLM 을 만들어보고 싶다. 개념부터 배우고 구현하고 싶다." (project-seed 메뉴 위저드)
- **결정**: 이름 `sh-llm-study`. 범위 = 개념 학습 + 밑바닥부터 소형 한글 GPT (CPU 전용). 스택 Python 3.12 + PyTorch CPU
  + Jupyter, uv. 저장은 파일만, `data/` → `/mnt/d/sh-llm-data`(D 드라이브, 탐색기로 보기 위함). 코퍼스 한글 퍼블릭 도메인.
  대시보드 FastAPI + React + Tailwind, 디자인 Linear(oh-my-design), 마지막 마일스톤. 상세 `docs/adr/0002-stack.md`.
- **만든 것**: 템플릿 인스턴스화, `docs/brief.md`(draft), ADR-0002, `docs/plans/0001-mvp.md`(M0~M6, 0~10장),
  `pyproject.toml`(uv, torch 2.14+cpu), `src/shllm/{config,data,tokenizer}.py`, 테스트 4개, `scripts/download_corpus.py`
  (위키문헌 20편 → 114만 자, 머리말/라이선스 상자 제거), `scripts/verify.sh`, `scripts/hooks/format.sh`(ruff),
  0장 교재 `docs/book/00-orientation.md` + `notebooks/00-orientation.ipynb`, nbstripout git 필터.
- **검증**: `bash scripts/verify.sh` 통과 (ruff · pytest 4 passed · 노트북 00 실행 6초). 2048² 행렬곱 73 GFLOPS.
- **환경 메모**: uv 는 이번에 설치(`~/.local/bin/uv`). `docker` 는 WSL 통합이 꺼져 있어 M6 전에 Docker Desktop 설정 필요.
  node 는 volta 경유로 동작 확인.
- 브리프 사용자 확인 완료(`status: confirmed`), 피드백 채널 = 채팅 + `feedback/` 폴더.
- **다음**: M1 1장(문자 토크나이저 + 바이그램) 착수.
