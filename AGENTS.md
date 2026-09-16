# AGENTS.md — sh-llm-study

이 파일은 **모든 AI 코딩 에이전트**(Claude Code, Codex, Cursor, Copilot 등)를 위한 진입점이다.
도구에 상관없이 아래 규칙을 따른다. Claude Code 전용 상세(부팅 프로토콜·하네스)는 `CLAUDE.md`.

## 이 프로젝트가 무엇인가

LLM 개념을 10개 장으로 배우며 CPU 만으로 소형 한글 GPT 를 밑바닥부터 구현·학습하는 학습 프로젝트.
스택: Python 3.12 + PyTorch(CPU) + Jupyter, 패키지 관리 uv, 테스트 pytest, 린트 ruff. 코퍼스는 한국 근대문학
퍼블릭 도메인 20편. 대시보드(마지막 마일스톤)는 FastAPI + React(Vite·TS) + Tailwind, 디자인 레퍼런스 Linear.
데이터는 `data/` → `/mnt/d/sh-llm-data`. 상세: `docs/brief.md`, `docs/adr/0002-stack.md`, 로드맵 `docs/plans/0001-mvp.md`.

## 절대 규칙 (위반 금지)

1. **작업 전에 읽는다**: `docs/session-log.md`(진행 이력 SSOT) → `docs/todo.md` → 최근 `docs/plans/*` 1개.
2. **코드를 바꾸면 같은 턴에 문서를 갱신한다**: session-log 최상단 블록 + todo. 규칙은 `docs/writing-guide.md`.
3. **검증 없이 전달하지 않는다**: `bash scripts/verify.sh` 통과가 커밋 메시지 작성의 전제.
4. **`.env` 를 직접 수정하지 않는다.** `.env.example` 수정 또는 사용자에게 요청.
5. **`git add -A` / `git add .` 금지.** 파일을 지정해서 stage 한다. 커밋·푸시는 verify 통과 후 에이전트가 수행(사용자 위임).
6. **스택·구조 변경은 ADR 로**: 구조적 결정은 `docs/adr/NNNN-*.md` 에 근거와 함께 기록한다.
7. **버전의 MINOR 승격은 사용자 선언이 있을 때만.**
8. **브리프가 판정 기준**: 설계 논쟁·스코프 논쟁은 `docs/brief.md` 로 돌아와 판정한다.

## 커밋 컨벤션

Conventional Commits — `<type>: <한국어 제목>` + 리스트형 본문 + `Co-Authored-By` 트레일러.
type: feat, fix, refactor, chore, docs, style, test, perf, ci, build, revert, init, remove, rename, hotfix.

## 더 읽을 것

- `CLAUDE.md` — 부팅 프로토콜, 변경 후 자동 규칙, 코드 지도, 하네스 상세
- `docs/brief.md` — 왜/무엇 SSOT · `docs/writing-guide.md` — 문서 규칙
