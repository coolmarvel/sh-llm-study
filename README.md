# sh-llm-study

LLM 개념을 챕터별로 배우면서, **CPU 만으로 소형 한글 GPT 를 밑바닥부터 직접 구현·학습**하는 학습 프로젝트.

- 교재: `docs/book/00~10-*.md` 11장 → `scripts/build_book.py` 가 PDF 한 권으로
- 실습: `notebooks/NN-*.ipynb`
- 코드: `src/shllm/` — 토크나이저 → 임베딩 → autograd/MLP → 어텐션 → GPT → 학습 → 생성 → 평가 → SFT
- 대시보드: `dashboard/` — 손실 곡선·어텐션 맵·토큰별 생성 확률 (`docker compose up --build -d` → http://localhost:8082)
- 데이터: `data/` → `D:\sh-llm-data` (코퍼스·토크나이저·체크포인트·실험 로그)

## 시작하기

```bash
uv sync                                     # Python 환경 복원 (PyTorch CPU 포함)
uv run python scripts/download_corpus.py    # 한글 퍼블릭 도메인 코퍼스 20편 → data/corpus/
uv run jupyter lab                          # notebooks/00-orientation.ipynb 부터
```

학습·생성:

```bash
uv run python scripts/train.py --config configs/small-cpu.yaml      # 6.8M GPT, CPU 약 1시간 → data/checkpoints/small-cpu/best.pt
uv run python -c "from shllm.train import load_checkpoint; from shllm.tokenizer import BPETokenizer; from shllm.config import TOKENIZER_DIR; from shllm.generate import generate_text; m,_=load_checkpoint('small-cpu'); t=BPETokenizer.load(TOKENIZER_DIR/'bpe-8192.json'); print(generate_text(m,t,'옛날 옛적에',80,temperature=0.8,top_k=50))"
```

검증: `bash scripts/verify.sh` (ruff · pytest · 노트북 정규화 검사 · 노트북 전체 실행)

## 문서

- `docs/brief.md` — 왜/무엇
- `docs/plans/0001-mvp.md` — 챕터 로드맵
- `docs/adr/0002-stack.md` — 스택·저장·디자인 결정

---

© 2026 이성현 (SeongHyun Lee). All rights reserved. — `LICENSE`
