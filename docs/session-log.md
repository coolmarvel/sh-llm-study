---
title: 세션 로그
created: 2026-09-16
updated: 2026-09-16
domain: development
---

# 세션 로그 (최신이 위)

이 파일이 **"언제 무슨 일이 있었나"의 SSOT**다. 세션마다 최상단에 블록 추가.
(커밋/푸시는 사용자가 직접·성긴 단위, git history 를 이력 SSOT 로 삼지 않는다.)

블록 형식: `## YYYY-MM-DD, 제목` 아래에 **요청/피드백 → 수정 → 검증 → 다음** 순서로 간결하게.

## 2026-09-16 (오후) v1.1: 코퍼스 확장·웹 내장 노트북 실행기·교재 보강·줄표 제거

- **요청**: "다음 todo 를 네 판단으로 설계·진행" → plans/0002, ADR-0003. 이어서 피드백 넷:
  ① "교재·노트북이 너무 당연하다는 듯이만 적혀 있다, 더 상세하게" ② "JupyterLab 을 따로 띄우지 말고 웹에서 실행" ③ "교재 UI 가 구리다"
  ④ "줄표 기호 쓰지 마라. 어디서든 다".
- **코퍼스 2차 확장** (ADR-0003): `scripts/download_wiki.py` 위키 알찬·좋은 글 401편 + 링크 이웃 → 673편 1,000만 자 (CC BY-SA, `WIKI-LICENSE.md`).
  `korean-mixed.txt` 1,110만 자 → `bpe-8192-mixed.json` (624만 토큰, 글자당 1.79). `configs/mixed-cpu.yaml` (6.8M, 5,000스텝, dropout 0.1).
  `TrainConfig.corpus` 필드, `scripts/train_tokenizer.py`.
- **웹 내장 노트북 실행기**: `dashboard/api/kernels.py` (jupyter_client 로 노트북별 ipykernel, NDJSON 출력 스트림, 중단·재시작), 셀 읽기/저장 API,
  `/api/source` 소스 뷰어. 웹은 CodeMirror 편집기 + 출력(텍스트·이미지·에러) + 모두 실행/중단/재시작/저장. **`lab` 서비스 제거** (컨테이너 하나).
- **교재 UI**: 챕터 목록(읽음 표시)·절 목차(스크롤 추적)·이전/다음·코드 경로 클릭 → 소스 패널·콜아웃. 실험 탭 run 겹쳐 그리기, 생성 탭 스트리밍.
- **교재·노트북 보강**: 0~10장 전부에 읽기 전에·핵심 용어·손계산 단계별 예시·자바 개발자의 눈으로·흔한 오해·스스로 점검 절 추가.
  노트북마다 "출력에서 볼 것"·"해 보기" 셀. 규칙은 `docs/writing-guide.md` "상세함의 기준".
- **줄표 제거**: 저장소 전체 600여 건 (제목 `1장 …`, 절 `1.1 …: …`, 본문 쉼표·마침표). 규칙은 writing-guide "언어" 절 + 에이전트 메모리.
- **사고**: 대기 스크립트의 `pgrep -f "docker compose build"` 가 자기 명령줄에 걸려 무한 대기 → 파일 마커·포트로 기다리는 규칙(CLAUDE.md 함정).
- **mixed-cpu 결과**: 5,000 스텝 113분, val(혼합) 4.436 @5000 (끝까지 하강, 과적합 없음). 근대문학 val 글자당 **3.30 bit** (small-cpu 3.48). 위키식 프롬프트 이어쓰기 가능. 9장 표·8장 §5·7장 7.5 에 반영.
- **검증**: `bash scripts/verify.sh` 통과 (ruff · pytest 61 · 노트북 11개 실행, 보강본). PDF v0.1.5 바탕화면 교체. 버전 0.1.5. 커밋 4묶음·푸시.
- **메모**: 하네스가 "메모리 부족"으로 백그라운드 대기 작업을 3회 종료 (free 1.3GB, available 4.3GB). 긴 대기는 `Monitor` 로.

## 2026-09-16: 4~10장 + 본 학습 + 대시보드(M6) 일괄 구현

- **요청**: "4장부터 끝까지, todo 까지 다 구현. 장별로 테스트도." (한 번에)
- **만든 것** (장별 교재·노트북·모듈·테스트 7종 세트):
  - 4장 `autograd.py`(Value)·`numpy_lm.py`(손 역전파)·`mlp.py`(MLP LM)·`data.get_batch`. NumPy 바이그램 val 3.58(세기 3.10), MLP T=1 val 6.85 < 2-gram 7.67
  - 5장 `attention.py`, 문맥 8: 어텐션 한 층 val 6.40 vs MLP 8.22
  - 6장 `model.py`(GPTConfig·Block·GPT, weight tying), 2층·C128 600스텝 val 6.1
  - 7장 `train.py`(Trainer·스케줄·체크포인트·JSONL)·`scripts/train.py`·`configs/{tiny-notebook,small-cpu}.yaml`, **small-cpu 본 학습** 6.8M, 2,500스텝 (결과는 아래)
  - 8장 `generate.py`(temperature·top-k·top-p·반복 억제, 원본/조정 확률) · 9장 `eval.py`(perplexity·bits/char·scaling_experiment) · 10장 `sft.py`(작품 목록 질문-답 80개 SFT 시연, 환각 시연)
  - `data.WORKS` 를 작품 목록 SSOT 로 승격 (download_corpus 가 import)
  - **M6 대시보드**: `dashboard/api/main.py`(FastAPI: runs·log·attention·generate·tokenize) + `dashboard/web`(Vite·React·TS·Tailwind v4·Recharts, 화면 3개) + `dashboard/Dockerfile`·`docker-compose.yml`(8082, D 드라이브 :ro) + `DESIGN.md`(Linear 파생) + `docs/guides/{dashboard,training}.md`. Playwright 1360/400 뷰포트 확인.
  - oh-my-design 스킬 번들 설치(`.claude/skills/omd-*`, agents, hooks). 1장 교재에 스무딩 α·Kneser-Ney 절 추가(P4).
  - **사용자 요청(채팅)**: "docs 와 notebooks 도 사이트에서 보고, 노트북은 실행해서 결과도 보고 싶다" → 대시보드에 **교재 탭**(`/api/book`, markdown→HTML)·
    **노트북 탭**(`/api/notebooks`, verify.sh 실행본 `build/notebooks/` 를 nbconvert HTML 로 + JupyterLab 열기 버튼) 추가, 컴포즈에 **`lab` 서비스**(JupyterLab 8888,
    Dockerfile target `lab`, Noto CJK 폰트, src/notebooks/docs/configs/build 마운트, D 드라이브 rw) 추가. 가이드·CLAUDE.md 갱신.
- **사고·결정 (근거 1줄씩)**:
  - `setup_cpu()` 기본 16 스레드 → MLP 스텝 275ms, 8 스레드 17ms (**16배**). 기본을 물리 코어(`PHYSICAL_CORES`)로 변경.
  - 4장 검증 loss 를 val 전체(11만×2,827 softmax = 1.3GB)로 계산해 스왑 → 2만 토큰 / 배치 평균으로.
  - small-cpu 초안(문맥 256, 6,000스텝)은 스텝당 5초·10시간 + 코퍼스 49만 토큰이라 과적합 확정 → 문맥 128·dropout 0.2·2,500스텝(약 1시간).
  - `omd install-skills` 가 `.claude/settings.json` 의 Remote Control 차단 키를 삭제 → 즉시 복원. CLAUDE.md 에 경고.
  - Dockerfile `COPY a b ./dir/` 는 b 디렉토리 **내용**을 푼다 → `dashboard.api` import 실패 → 경로 분리.
  - Playwright MCP 는 `pkill -f uvicorn` 문자열이 자기 명령줄에도 걸려 대기 작업이 죽음 → 이후 `pkill` 패턴 주의.
  - DESIGN.md 는 `/omd:init` 의 hash-bound 패키지 대신 카탈로그에서 손으로 파생 (스킬은 새 세션에서만 로드됨), todo P3.
- **본 학습 결과 (small-cpu)**: 2,500 스텝 45분, best val **5.485** @2100 (train 4.77), 마지막 5.49, 2,100 이후 정체(과적합 시작). perplexity 241, 글자당 3.48 bit (문자 3-gram 4.15). 생성문은 어절·대화 부호는 살아 있으나 문단 의미는 이어지지 않음 (v1.0 판정은 사용자, todo P1).
- **검증**: `bash scripts/verify.sh` 통과 (ruff · pytest 58 · 노트북 정규화 · 노트북 11개 실행 ~30분). PDF v0.1.4(11장) 바탕화면 교체. 버전 0.1.4. 장별 커밋·푸시.
- **다음**: 사용자 v1.0 판정 → 코퍼스 2차 확장 (todo P1).

## 2026-09-16: 사고: VS Code 에서 노트북이 열자마자 dirty (03)

- **피드백(채팅)**: "03 을 열면 변경사항이 자꾸 생겨 바로 닫기가 안 된다. 다른 노트북은 괜찮다."
- **원인**: VS Code Jupyter 확장이 .venv 커널을 고르며 `metadata.kernelspec.display_name = "sh-llm-study (3.12.3)"` 과
  `metadata.language_info` 를 써 넣는데, nbformat 으로 생성한 노트북에는 이 값이 없어 열 때마다 변경이 생김. (사용자가 Ctrl+S 한 파일을
  HEAD 와 비교해 확인.) 03 만 문제였던 것은 에이전트가 같은 시간대에 03 을 여러 번 덮어써 dirty 버퍼가 남았기 때문으로 추정.
- **수정**: `scripts/normalize_notebooks.py`, 표준 메타데이터·순번 셀 id·출력 제거를 제자리 적용, `--check` 를 `scripts/verify.sh` 에 추가
  (어긋나면 검증 실패). 노트북 4개 정규화. 노트북 빌더는 이후 이 스크립트를 마지막에 호출한다.
- **규칙**: 사용자가 열어 둘 수 있는 노트북을 덮어쓰기 전에 한 줄 알린다 (에이전트 메모리에도 기록).

## 2026-09-16: M2 3장 임베딩

- **요청**: "Untitled.ipynb 삭제해도 되고, 다음 스텝으로" → 삭제 + `.gitignore` 에 `Untitled*.ipynb`.
- **만든 것**: `src/shllm/embedding.py`(`TokenEmbedding`·`one_hot_lookup`·`sinusoidal_positions`·`LearnedPositionalEmbedding`·
  `GPTEmbedding`, 학습 없이 벡터를 만드는 `cooccurrence_matrix`·`ppmi`·`svd_embeddings`·`nearest`), `tests/test_embedding.py`(7개),
  `docs/book/03-embedding.md`, `notebooks/03-embedding.ipynb`(실행 ~30초). 버전 0.1.3.
- **설계 결정**: 3장은 4장(autograd) 앞이라 "학습된 벡터"를 만들 수 없음 → 동시출현(±2칸, 상위 4,000 토큰)+PPMI+SVD 로 세어서
  만든 벡터로 이웃(` 아버지`→` 어머니`, ` 돈`→` 원`·` 이천`, ` 없다`→` 없었다`)을 보여 "1장 확률표의 압축 = 임베딩" 감각을 만든다.
  학습된 임베딩 시각화는 7장 뒤로 (todo).
- **검증**: `bash scripts/verify.sh` 통과, PDF 재빌드(v0.1.3), 커밋·푸시.
- **다음**: M2 4장 신경망 기초.

## 2026-09-16: M1 2장 BPE 토크나이저 + 교재 PDF 빌드 + 공개 저장소

- **요청**: "다음 스텝 진행, build_book.py 만들어 바탕화면에 복사·갱신 시 이전 PDF 교체, GitHub public 생성·푸시" + 질문 "만들면 대화가 되나?" (→ 채팅으로 답: base 모델은 이어쓰기만, 대화는 SFT 필요. todo P2 에 10장 스코프 항목).
- **만든 것**: `scripts/build_book.py`(weasyprint, `build/book/sh-llm-study-book-v<버전>.pdf` → 바탕화면 이전 판 삭제 후 복사),
  CLAUDE.md 규칙(교재 변경 시 같은 턴 재빌드, 체크리스트 ⑦), GitHub `coolmarvel/sh-llm-study`(public) 생성·푸시.
  `BPETokenizer`(`src/shllm/tokenizer.py`: 바이트 BPE + 증분 학습 + `char_first` 글자 조립 + `truncated`), 테스트 7개,
  `docs/book/02-bpe-tokenizer.md`, `notebooks/02-bpe-tokenizer.ipynb`(실행 ~3분), `data/tokenizers/bpe-8192.json`. 버전 0.1.2.
- **설계 결정(사고 기록)**: 순수 바이트 BPE 는 한글에서 글자 경계를 걸치는 조각이 어휘의 1/6 을 차지("옛" 이 두 조각) →
  `char_first=True` 로 2회 이상 나온 글자를 먼저 조립(2,520 병합). 압축률은 같으나(8192 에서 2.10자/토큰) 토큰이 온전한 글자 단위.
  단순 병합(매번 전체 재계산)은 8천 병합에 25분 → `pair_counts`+`pair_words` 증분 갱신으로 ~1.5분.
- **검증**: `bash scripts/verify.sh` 통과, PDF 재빌드·바탕화면 교체, 커밋·푸시.
- **다음**: M2 3장 임베딩.

## 2026-09-16: M1 1장: 문자 토크나이저 + 바이그램/n-gram

- **요청**: "진행해줘" (1장 착수). Docker Desktop 실행 확인 요청 → `docker` 29.7 동작 확인, todo P2/plan 항목 해소.
- **만든 것**: `src/shllm/bigram.py`(`BigramModel` (V,V) 텐서 + `NGramModel` dict 보간), `tests/test_bigram.py`(6개),
  `docs/book/01-char-tokenizer-bigram.md`, `notebooks/01-char-tokenizer-bigram.ipynb`(31셀, 실행 ~50초),
  `data/tokenizers/char.json`(V=2,827). 버전 0.1.0 → 0.1.1 (PATCH).
- **설계 결정(사고 기록)**: 라플라스 스무딩 α=1 은 V=2,827 에서 실제 횟수를 묻어 생성이 한자 범벅이 됨 → `BigramModel`
  기본 α=0.01. n≥3 은 균등분포 스무딩으로는 문맥 하나만 못 봐도 생성이 즉시 무너짐 → `NGramModel` 은 한 단계 짧은
  문맥 분포로 **보간**(재귀). 결과 val loss: 균등 7.95 / bigram 3.17 / 3-gram 2.88(최소) / 5-gram 3.50(과적합).
- **검증**: `bash scripts/verify.sh` 통과 (ruff · pytest 10 passed · 노트북 00·01 실행). 로컬 커밋 완료.
- **푸시 보류**: git 원격이 없고 GitHub 에 `coolmarvel/sh-llm-study` 도 없음. 저장소 생성(공개/비공개)은 사용자 결정 → 생성 후 `git remote add origin … && git push -u origin main`.
- **미처리**: 루트 `Untitled.ipynb`(빈 노트북, Jupyter 가 만든 것으로 추정) 은 건드리지 않음. 사용자 확인 필요.
- **다음**: M1 2장 BPE 토크나이저 (`BPETokenizer`, `data/tokenizers/` 저장 규약).

## 2026-09-16: 킥오프 완료

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
