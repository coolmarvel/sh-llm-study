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

- [ ] **M2 4장** 신경망 기초 — autograd·경사하강 NumPy 재현 → PyTorch MLP 언어모델 (Bengio 2003), `src/shllm/mlp.py`(가칭) (`docs/plans/0001-mvp.md`)
- [ ] 7장 학습 후 3장 노트북의 PCA 그림을 학습된 토큰 임베딩으로 다시 그리기 (`notebooks/03-embedding.ipynb` §2 참조)

## P2 — 가까운 로드맵

- [ ] M2~M5 장들 (`docs/plans/0001-mvp.md`)
- [ ] M6 착수 시: `npx oh-my-design-cli@latest` 설치 + `/omd:init Linear` → `DESIGN.md` (CLAUDE.md 디자인 절)
- [ ] 코퍼스 2차 확장(한국어 위키 일부) 여부 — 7장 학습 결과 보고 결정
- [ ] 프로젝트 어휘 크기 확정 — 지금은 `bpe-8192.json`(2.10자/토큰). 9장에서 4096/8192 손실 비교 후 결정. 토크나이저 바꾸면 체크포인트 호환 깨짐
- [ ] 10장 스코프: 사용자 최종 기대는 "대화" — 미니 SFT(대화 형식 파인튜닝) 데모를 넣을지 사용자와 결정 (2026-09-16 질문)

## P3 — 품질

- [ ] `scripts/download_corpus.py` 의 `drop_boilerplate` 는 위키문헌 페이지 구조에 의존 — 페이지가 바뀌면 깨질 수 있음. 합본에 남은 잡음 문구 4건 확인

- [ ] `NGramModel.loss` 는 파이썬 루프(1.1M 토큰 × n 회 dict 조회, n=5 에 ~6초) — 9장 평가에서 재사용하면 벡터화 검토

## P4 — 아이디어

- [ ] 1장 확장: n-gram 보간 α 를 val 로 고르는 실험 (α=3 이 n≥3 에서 val 더 낮음 — 세션 실험 메모), Kneser-Ney 소개
