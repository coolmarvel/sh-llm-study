# CLAUDE.md — sh-llm-study 작업 가이드

이 파일은 **세션이 바뀌어도 맥락을 즉시 복구**하기 위한 진입점이다. Claude Code는 세션 시작 시
이 파일을 자동으로 읽는다. (도구 중립 절대 규칙은 `AGENTS.md`.)

> [project-seed](https://github.com/coolmarvel/project-seed) 에서 2026-09-16 에 생성됨.
> `<!-- TODO(kickoff) -->` 가 남아 있으면 킥오프가 끝나지 않은 것이다 — 채우기 전에 기능 작업을 시작하지 않는다.

## 🟢 세션 시작 부팅 프로토콜 (매 세션 첫 작업 전에 반드시 수행)

새 세션에서 작업을 시작하면, **코드를 건드리기 전에** 순서대로:

1. `docs/session-log.md` 읽기 — 마지막으로 무엇을 했고 지금 어디쯤인지 (**진행 이력 SSOT**)
2. `docs/todo.md` 읽기 — 남은 일 (P1~P4)
3. 최근 `docs/plans/*.md` 1개 읽기 — 진행 중 기능의 설계 의도
4. 사용자 피드백 확인: 루트 `feedback/` 폴더의 스크린샷·메모 = 미처리 피드백. 반영 후 `docs/feedback-archive/YYYY-MM-DD-*/` 로 이동. 채팅으로 온 피드백은 session-log 블록에 요약.
5. `git status` 에 모르는 변경이 있으면 session-log 와 대조 — 다른 세션의 흔적일 수 있다. 출처 불명이면 사용자에게 확인.

## 🔴 변경 후 자동 규칙 (사용자가 매번 요청하지 않아도 수행)

1. 코드를 바꾸면 **같은 턴에** `docs/session-log.md`(최상단 블록 추가)·`docs/todo.md`
   (+릴리스급이면 `docs/changelog.md`)를 갱신한다. 문서 규칙은 `docs/writing-guide.md`.
2. 검증을 통과하기 전에는 커밋 메시지 작성/산출물 전달을 하지 않는다:
   `bash scripts/verify.sh` (ruff format --check · ruff check · pytest · `notebooks/*.ipynb` 전체 실행)
3. 버전을 판단해 올린다 (아래 "버전 정책").
4. 산출물 전달: 챕터 = 교재(`docs/book/`)+노트북+`src/shllm/` 모듈이 verify 통과한 상태.
   **`docs/book/` 이 바뀌면 같은 턴에 `uv run python scripts/build_book.py`** — 바탕화면(`/mnt/c/Users/user/Desktop/`)의
   이전 판 `sh-llm-study-book-*.pdf` 를 지우고 새 판을 복사한다 (항상 최신 한 권만, 사용자 지시 2026-09-16).
   학습된 모델 = `data/checkpoints/<run>/` (D 드라이브, 탐색기로 확인 가능). 대시보드 = 도커 8082 포트에서 사용자와 함께 확인.

## 버전 정책 (semver `MAJOR.MINOR.PATCH`)

**MINOR 승격은 사용자만 선언한다.** 에이전트가 판단해서 올리지 않는다.

- **PATCH** — 기본값. 다듬는 중인 기능의 수정 하나 반영할 때마다 +1.
- **MINOR** — 사용자가 "이 기능은 더 수정할 게 없다, 넘어가자"라고 선언한 그 시점에만.
- **MAJOR** — 대규모 재설계/호환 깨짐. 사용자와 상의.

## 커밋 컨벤션

Conventional Commits — `<type>: <한국어 제목>` + 리스트형 본문. **검증(verify.sh) 통과 후 에이전트가 직접 커밋·푸시한다**
(사용자 위임, 2026-09-16). 원격: `github.com/coolmarvel/sh-llm-study` (public). 저자는 `coolmarvel <marvel97@naver.com>`. 메시지 끝에 `Co-Authored-By` 트레일러.

type: `feat` `fix` `refactor` `chore` `docs` `style` `test` `perf` `ci` `build` `revert` `init` `remove` `rename` `hotfix`

## 협업 규칙

- **진행 이력의 SSOT 는 `docs/session-log.md`** — git history 가 아니다 (커밋이 성긴 단위라서).
- 세션이 끊겨도 이어서 작업할 수 있게 **모든 진행 사항을 `docs/` 에 파일로 기록**한다.
- 설계 논쟁이 생기면 `docs/brief.md`(왜/무엇 SSOT)로 돌아와 판정한다. 브리프 밖 기능은 스코프 확인 먼저.
- `.env` 는 직접 수정하지 않는다 — `.env.example` 수정 또는 사용자에게 요청 (훅이 차단함).
- 규칙이 미확정이면 이 파일에 `<!-- TODO -->` 로 남기고, 확정되는 순간 채운다.

## 이 프로젝트가 뭔가

LLM 을 처음 배우는 자바 배경 개발자(제작자 본인)가, **개념을 10개 장으로 익히면서 CPU 만으로 소형 한글 GPT 를
밑바닥부터 구현·학습**하는 학습 프로젝트. 한 장 = 교재 문서 + 노트북 + `src/shllm/` 모듈 + 테스트.
코퍼스는 위키문헌의 한국 근대문학 20편(퍼블릭 도메인, 114만 자). v1.0 기준은 "한글 프롬프트를 이어서 말이
되는 문단을 생성 + 모든 장을 내 말로 설명할 수 있다". 벤치마킹: nanoGPT / "Build a LLM from scratch" 의 구조를
한글·CPU·자바 개발자 관점으로 다시 쓴 것. 상세는 `docs/brief.md`, 로드맵은 `docs/plans/0001-mvp.md`.

**학습 프로젝트 특유의 규칙**: 코드는 "가장 짧은 정답"보다 **"읽으면서 배우는 코드"** 를 우선한다 — shape 주석
`(B, T, C)`, 한글 docstring, 자바 개발자가 낯설 파이썬 관용구에는 한 줄 설명. 노트북 셀 하나는 10분 이내.

## 문서 인덱스 (docs/)

| 파일 | 용도 |
|---|---|
| `docs/writing-guide.md` | **문서 지배 규칙** (SSOT·frontmatter·코드 1:1 대조). 문서 쓰기 전 필독 |
| `docs/brief.md` | 프로젝트 **왜/무엇 SSOT** — 킥오프 산출물 |
| `docs/session-log.md` | 세션별 진행 이력. **"언제 무슨 일" SSOT** (최신이 위) |
| `docs/todo.md` | 미해결·향후 작업만 (P1~P4). 완료분은 session-log 로 |
| `docs/changelog.md` | 릴리스 단위 사람용 요약 |
| `docs/adr/*.md` | 구조적 결정 기록 — 왜 이렇게 했는가 (`NNNN-kebab.md`) |
| `docs/plans/*.md` | 앞으로 만들 것 — 기능 단위 구현 계획 (역할 구분은 `plans/README.md`) |
| `docs/guides/*.md` | 현재 구현된 동작·코드 위치 (기능별) |
| `docs/book/NN-*.md` | **교재** — 장별 개념 설명. 노트북과 1:1. `scripts/build_book.py` 가 PDF 로 묶음 |
| `docs/feedback-archive/` | 처리 완료한 사용자 피드백 보관소 |

## 자주 쓰는 명령

```bash
uv sync                                        # 환경 복원 (uv 는 ~/.local/bin/uv)
uv run python scripts/download_corpus.py       # 코퍼스 → data/corpus/ (이미 있으면 skip, --force 로 재수집)
uv run jupyter lab                             # 노트북 실습
uv run pytest                                  # 단위 테스트
uv run ruff format . && uv run ruff check .    # 포맷·린트 (Write/Edit 훅이 .py 는 자동 포맷)
bash scripts/verify.sh                         # 검증 전체 (커밋 전 필수)
# 긴 학습(7장 이후): uv run python scripts/train.py --config configs/<이름>.yaml
# 교재 PDF:         uv run python scripts/build_book.py   # build/book/ + 바탕화면 교체 (--no-copy 로 굽기만)
# 대시보드(M6):     docker compose up  → http://localhost:8082
```

노트북은 `nbstripout` git 필터가 출력을 제거하고 커밋한다 (`uv run nbstripout --install` 이 clone 마다 필요).
노트북을 새로 만들거나 재생성한 뒤에는 `uv run python scripts/normalize_notebooks.py` — VS Code 커널 메타데이터를 미리 넣어 열자마자 dirty 가 되지 않게 한다 (verify.sh 가 `--check`).

## 코드 지도 (수정 시 어디를 보나)

| 위치 | 역할 |
|---|---|
| `src/shllm/config.py` | 경로 SSOT(`DATA_DIR`·`CORPUS_DIR`·`CHECKPOINT_DIR`…) + `setup_cpu()`. 경로를 다른 곳에 하드코딩하지 않는다 |
| `src/shllm/data.py` | 코퍼스 읽기 (`load_corpus`, `list_work_files`) |
| `src/shllm/tokenizer.py` | 1장 `CharTokenizer` → 2장 `BPETokenizer` |
| `src/shllm/{embedding,attention,model,train,generate}.py` | 3·5·6·7·8장에서 생성 (아직 없음) |
| `notebooks/NN-*.ipynb` | 장별 실습. 검증된 코드는 반드시 `src/shllm/` 로 옮기고 노트북은 import 해서 쓴다 |
| `docs/book/NN-*.md` | 장별 교재. 노트북과 번호·제목이 1:1 |
| `tests/test_*.py` | 모듈별 단위 테스트 (round-trip, shape, 작은 학습이 손실을 줄이는지) |
| `scripts/` | `download_corpus.py`(코퍼스) · `verify.sh`(검증) · `build_book.py`(교재 PDF) · `normalize_notebooks.py`(노트북 메타데이터 표준화 — 새 노트북 저장 후 실행) · `hooks/`(하네스) · 이후 `train.py` |
| `configs/` | 학습 설정 YAML (7장 이후) |
| `data/` → `/mnt/d/sh-llm-data` | corpus/ tokenizers/ checkpoints/ runs/ — git 에 안 들어감 |

**새 장 추가 = ① `docs/book/NN-제목.md` ② `notebooks/NN-제목.ipynb` ③ `src/shllm/모듈.py` ④ `tests/test_모듈.py`
⑤ `docs/plans/0001-mvp.md` 체크 ⑥ `bash scripts/verify.sh` ⑦ `uv run python scripts/build_book.py`(PDF 교체)** — 일곱 개가 다 있어야 한 장이 끝난 것이다.

함정:
- 코퍼스 합본(`korean-classics.txt`)과 `works/*.txt` 를 같이 읽으면 데이터가 2배로 중복된다 → `load_corpus("korean-classics")` 만 학습에 쓴다.
- `/mnt/d` 는 느리다 → 학습 루프 안에서 D 드라이브를 반복 읽지 않는다. 코퍼스는 시작 시 한 번 메모리로.

## 디자인 시스템 (oh-my-design)

대시보드(M6, `dashboard/`)에만 적용. 챕터 노트북·교재에는 해당 없음.

디자인 레퍼런스: **Linear** (베이스), 차용 없음 — ADR-0002. **M6 착수 시** `npx oh-my-design-cli@latest` 설치 후
`/omd:init Linear` 로 `DESIGN.md` 를 만든다 (킥오프 시점엔 미설치 — todo P2).
디자인 계약은 루트 **`DESIGN.md`** (프로젝트 소유, 커밋 대상). 색·간격·컴포넌트 구조는 여기서만 정하고
코드에 임의 값을 넣지 않는다. 설치 확인: `npx oh-my-design-cli@latest doctor`.

| 상황 | 반드시 쓰는 명령 |
|---|---|
| 새 화면·기능 UI 만들기 | `/omd:harness` (조사→설계→구현→검증, 체크포인트 3개) — 작은 컴포넌트 하나면 `/omd:apply` |
| 기존 UI 손보기 | `/omd:apply` 로 `DESIGN.md` 기준 수정 → `/omd:slop-audit` |
| UI 작업 마무리 전 | `/omd:slop-audit`(AI 티·일관성 감사) + `/omd:feel`(사용감 점검) 통과 후에만 산출물 전달 |
| 디자인 원칙·토큰 바꾸기 | `/omd:init` 재실행 또는 `DESIGN.md` 직접 수정 → 변경 이유를 `docs/adr/` 에 기록 |
| 디자인 문서 열람 | `npx oh-my-design-cli@latest book` (localhost:6060) |
| 번들 갱신 | `npx oh-my-design-cli@latest update` 후 `doctor` |

## 하네스 (Harness Engineering)

지시문이 아니라 시스템으로 제약한다. 도입 근거: `docs/adr/0001-harness-engineering.md`.

| 축 | 내용 |
|---|---|
| Hooks (`scripts/hooks/`) | `env-guard.sh`(.env 편집 차단) · `git-add-guard.sh`(`git add -A/.`·.env staging 차단) · `format.sh`(PostToolUse, `.py` 를 ruff format + check --fix) |
| Settings (`.claude/settings.json`) | `disableRemoteControl: true` + `remoteControlAtStartup: false` + `autoUploadSessions: false` — Claude Code **Remote Control**(로컬 세션을 claude.ai/code 에 동기화) 차단. 대화 내용이 웹에 남지 않게 한다. 세션 중 `/rc` 가 켜져 있으면 끊는다 |
| Commands (`.claude/commands/`) | `/review-security` · `/deploy-check` |
| MCP (`.mcp.json`) | `context7`(라이브러리 문서 — PyTorch API 확인에 사용) · `playwright`(M6 대시보드 QA) |
| Skills | oh-my-design 번들(M6 에서 설치) |
