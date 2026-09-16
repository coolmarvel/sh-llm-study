---
title: TODO
created: 2026-09-16
updated: 2026-09-16
domain: development
---

# TODO (미해결·향후 작업만 — 완료분은 session-log 로)

우선순위: **P1** 다음 릴리스에서 다뤄야 함 · **P2** 가까운 로드맵 · **P3** 품질 · **P4** 아이디어.
항목에는 대상 파일 경로와 (있다면) 과거 사고 근거를 함께 적는다.

## P1 — 다음 릴리스에서 다뤄야 함

- [ ] **M1 1장** 문자 토크나이저 + 바이그램 통계 모델 — `docs/book/01-*.md`, `notebooks/01-*.ipynb`, `src/shllm/tokenizer.py`(있음)·`bigram.py`, 테스트
- [ ] **M1 2장** BPE 토크나이저 — `src/shllm/tokenizer.py` 에 `BPETokenizer`, `data/tokenizers/` 저장 규약 (`docs/plans/0001-mvp.md`)

## P2 — 가까운 로드맵

- [ ] M2~M5 장들 (`docs/plans/0001-mvp.md`)
- [ ] M6 착수 전: Docker Desktop → Settings → Resources → WSL integration 켜기 (현재 `docker` 명령 불가)
- [ ] M6 착수 시: `npx oh-my-design-cli@latest` 설치 + `/omd:init Linear` → `DESIGN.md` (CLAUDE.md 디자인 절)
- [ ] 코퍼스 2차 확장(한국어 위키 일부) 여부 — 7장 학습 결과 보고 결정

## P3 — 품질

- [ ] `scripts/download_corpus.py` 의 `drop_boilerplate` 는 위키문헌 페이지 구조에 의존 — 페이지가 바뀌면 깨질 수 있음. 합본에 남은 잡음 문구 4건 확인

## P4 — 아이디어
