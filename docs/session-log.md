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

## 2026-09-16 — M1 1장: 문자 토크나이저 + 바이그램/n-gram

- **요청**: "진행해줘" (1장 착수). Docker Desktop 실행 확인 요청 → `docker` 29.7 동작 확인, todo P2/plan 항목 해소.
- **만든 것**: `src/shllm/bigram.py`(`BigramModel` (V,V) 텐서 + `NGramModel` dict 보간), `tests/test_bigram.py`(6개),
  `docs/book/01-char-tokenizer-bigram.md`, `notebooks/01-char-tokenizer-bigram.ipynb`(31셀, 실행 ~50초),
  `data/tokenizers/char.json`(V=2,827). 버전 0.1.0 → 0.1.1 (PATCH).
- **설계 결정(사고 기록)**: 라플라스 스무딩 α=1 은 V=2,827 에서 실제 횟수를 묻어 생성이 한자 범벅이 됨 → `BigramModel`
  기본 α=0.01. n≥3 은 균등분포 스무딩으로는 문맥 하나만 못 봐도 생성이 즉시 무너짐 → `NGramModel` 은 한 단계 짧은
  문맥 분포로 **보간**(재귀). 결과 val loss: 균등 7.95 / bigram 3.17 / 3-gram 2.88(최소) / 5-gram 3.50(과적합).
- **검증**: `bash scripts/verify.sh` 통과 (ruff · pytest 10 passed · 노트북 00·01 실행). 로컬 커밋 완료.
- **푸시 보류**: git 원격이 없고 GitHub 에 `coolmarvel/sh-llm-study` 도 없음. 저장소 생성(공개/비공개)은 사용자 결정 → 생성 후 `git remote add origin … && git push -u origin main`.
- **미처리**: 루트 `Untitled.ipynb`(빈 노트북, Jupyter 가 만든 것으로 추정) 은 건드리지 않음 — 사용자 확인 필요.
- **다음**: M1 2장 BPE 토크나이저 (`BPETokenizer`, `data/tokenizers/` 저장 규약).

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
