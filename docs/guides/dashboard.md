---
title: 대시보드 (M6)
created: 2026-09-16
updated: 2026-09-16
domain: dashboard
---

# 대시보드

## 개요

학습 로그·체크포인트를 읽어 **손실 곡선 · 어텐션 맵 · 토큰별 생성 확률**을 브라우저에서 보는 도구 (브리프 시나리오 6)에,
**교재 탭**(챕터·절 목차, 이전/다음. 코드 경로 클릭 → 소스 뷰어)과 **노트북 탭**(브라우저 안에서 셀 편집·실행·저장, 별도 Jupyter 서버 없음)을 더한 것.
FastAPI 가 `data/`·`docs/`·`notebooks/`·`build/notebooks/` 를 읽고, 모델을 메모리에 올리고, 노트북마다 ipykernel 을 띄워(`dashboard/api/kernels.py`) API 를 제공한다.
React(Vite·TS·Tailwind·CodeMirror) 가 다섯 화면을 그린다. 도커 컴포즈 서비스 하나(`dashboard`, 8082). 디자인 계약은 루트 `DESIGN.md` (Linear 톤, ADR-0002).

## 동작 방식

```
data/runs/<run>/log.jsonl ─┐
data/runs/<run>/config.yaml ├─▶ dashboard/api/main.py (FastAPI) ─▶ /api/* ─▶ dashboard/web (React) ─▶ 브라우저
data/checkpoints/<run>/best.pt ┘        └ load_checkpoint 로 run 당 한 번 로드 (lru_cache 4개)
```

| 엔드포인트 | 내용 | 화면 |
|---|---|---|
| `GET /api/runs` | run 목록: 모델·학습 설정, 파라미터 수(근사), step, best val, 체크포인트 유무 | 실험 |
| `GET /api/runs/{run}/log` | `log.jsonl` 그대로 | 실험(곡선) |
| `POST /api/attention {run, text}` | 층·헤드별 `(T, T)` 가중치 + 토큰 문자열 | 어텐션 |
| `POST /api/generate {run, prompt, temperature, top_k, top_p, repetition_penalty, max_new_tokens, top_n, seed}` | 자리마다 뽑힌 토큰·확률(조정 후/원본)·상위 후보 | 생성 |
| `POST /api/generate/stream` (같은 본문) | NDJSON: 첫 줄 `{prompt_tokens}`, 토큰마다 한 줄, 마지막 `{text}`, 생성 탭은 이것을 쓴다 | 생성 |
| `GET /api/tokenize?run=&text=` | 토큰 분해 |, |
| `GET /api/book` · `GET /api/book/{name}` | 챕터 목록 · markdown → HTML(+pygments css) | 교재 |
| `GET /api/notebooks` · `GET /api/notebooks/{name}/html?executed=` | 노트북 목록(실행본 유무) · nbconvert HTML(참고용) | 노트북 |
| `GET/PUT /api/notebooks/{name}/cells` | 셀 읽기(실행본 출력을 초기값으로) · 셀 소스 저장(출력 제외, 순번 id) | 노트북 |
| `POST /api/notebooks/{name}/execute {code}` | NDJSON 스트림: stream · display_data(image/png 등) · error · status(execution_count) | 노트북 |
| `POST /api/notebooks/{name}/kernel/{start,interrupt,restart,shutdown}` · `GET /api/kernels` | 커널 제어 (노트북마다 하나, 첫 실행 때 시작, 작업 폴더 `notebooks/`) | 노트북 |
| `GET /api/source?path=src/…` | 저장소 파일 하이라이트 HTML (`src/ scripts/ configs/ tests/ dashboard/ docs/` 만) | 교재 |
| `GET /{path}` | `dashboard/web/dist` 정적 파일 (빌드돼 있을 때만) |, |

조정 로직은 `src/shllm/generate.py` 의 것을 그대로 쓴다 (8장). 원본 확률(`raw_prob`)은 조정 전 softmax 라 "조정이 무엇을 바꿨나"를 화면에서 비교할 수 있다.

## 사용 방법

```bash
docker compose up --build -d            # http://localhost:8082
docker compose logs -f dashboard
docker compose down
```

- D 드라이브·`notebooks/` 는 읽기·쓰기(노트북 02·07 이 토크나이저·체크포인트를 쓰고, 셀 저장이 파일에 반영), `src/`·`docs/`·`configs/`·`build/notebooks/` 는 읽기 전용 마운트.
- 노트북 탭: 셀 실행은 서버가 띄운 ipykernel 에서 돈다(첫 실행 때 수 초). 상태(변수)는 커널 재시작 전까지 유지. `Shift+Enter` 로 실행, 설명 셀은 더블클릭으로 편집.
  "저장"은 셀 소스만 노트북 파일에 쓴다. 출력은 저장하지 않는다(git 규약과 동일). 실행본(`build/notebooks/`, verify.sh)이 있으면 그 출력이 초기값으로 보인다.
- 로컬 전용(인증 없음). 커널은 임의 코드를 실행하므로 8082 를 외부에 열지 않는다.

개발:

```bash
uv run uvicorn dashboard.api.main:app --port 8082 --reload   # API
cd dashboard/web && npm install && npm run dev               # http://localhost:5173, /api 는 8082 로 프록시
cd dashboard/web && npm run build                            # dist/ → FastAPI 가 서빙
```

체크포인트가 하나도 없으면 어텐션·생성 화면의 run 선택이 비어 있다. `scripts/train.py` 를 먼저 돌린다.

## 검증

- `tests/test_dashboard_api.py`, 임시 run·체크포인트로 4개 엔드포인트 (verify.sh 에 포함)
- Playwright(MCP)로 1360×860 · 400×800 두 뷰포트에서 세 화면 확인 (2026-09-16, `.playwright-mcp/shot-*.png`)

## 관련 코드

- `dashboard/api/main.py`, 엔드포인트 전부
- `dashboard/api/kernels.py`, `KernelPool` (jupyter_client 로 커널 시작·실행 스트림·중단·재시작)
- `dashboard/web/src/App.tsx` (셸·해시 라우팅) · `views/{Runs,Attention,Generate,Book,Notebooks}.tsx` · `components/{Heatmap,Markdown,SourcePanel}.tsx` · `api.ts` · `index.css` (DESIGN.md 토큰, `.prose-dark`, `.cell` 실행기 스타일)
- `dashboard/Dockerfile` (node 빌드 → python+uv+ipykernel, Noto CJK 폰트) · `docker-compose.yml`
- `DESIGN.md`, 토큰·컴포넌트 계약. 값은 여기서만 정한다
