# sh-llm-study

LLM 개념을 챕터별로 배우면서, **CPU 만으로 소형 한글 GPT 를 밑바닥부터 직접 구현·학습**하는 학습 프로젝트.

- 교재: `docs/book/NN-*.md` (마지막에 PDF 한 권으로 묶는다)
- 실습: `notebooks/NN-*.ipynb`
- 코드: `src/shllm/` — 챕터가 진행되며 토크나이저 → 임베딩 → 어텐션 → GPT → 학습 → 생성이 쌓인다
- 데이터: `data/` → `D:\sh-llm-data` (코퍼스·토크나이저·체크포인트·실험 로그)

## 시작하기

```bash
uv sync                                     # Python 환경 복원 (PyTorch CPU 포함)
uv run python scripts/download_corpus.py    # 한글 퍼블릭 도메인 코퍼스 20편 → data/corpus/
uv run jupyter lab                          # notebooks/00-orientation.ipynb 부터
```

검증: `bash scripts/verify.sh` (ruff · pytest · 노트북 전체 실행)

## 문서

- `docs/brief.md` — 왜/무엇
- `docs/plans/0001-mvp.md` — 챕터 로드맵
- `docs/adr/0002-stack.md` — 스택·저장·디자인 결정

---

© 2026 이성현 (SeongHyun Lee). All rights reserved. — `LICENSE`
