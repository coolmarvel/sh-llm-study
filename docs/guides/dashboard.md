---
title: 대시보드 (M6)
created: 2026-09-16
updated: 2026-09-16
domain: dashboard
---

# 대시보드

## 개요

학습 로그·체크포인트를 읽어 **손실 곡선 · 어텐션 맵 · 토큰별 생성 확률**을 브라우저에서 보는 도구 (브리프 시나리오 6)에,
**교재 탭**(`docs/book/*.md` 를 HTML 로)과 **노트북 탭**(실행 결과 HTML + JupyterLab 열기)을 더한 것.
FastAPI 가 `data/`·`docs/`·`notebooks/`·`build/notebooks/` 를 읽고 모델을 메모리에 올려 API 를 제공하고, React(Vite·TS·Tailwind) 가 다섯 화면을 그린다.
도커 컴포즈 서비스 둘: `dashboard`(8082, 웹+API) · `lab`(8888, JupyterLab — 노트북을 브라우저에서 실행). 디자인 계약은 루트 `DESIGN.md` (Linear 톤, ADR-0002).

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
| `GET /api/tokenize?run=&text=` | 토큰 분해 | — |
| `GET /api/book` · `GET /api/book/{name}` | 챕터 목록 · markdown → HTML(+pygments css) | 교재 |
| `GET /api/notebooks` · `GET /api/notebooks/{name}/html?executed=` | 노트북 목록(실행본 유무, lab URL) · nbconvert HTML | 노트북 |
| `GET /{path}` | `dashboard/web/dist` 정적 파일 (빌드돼 있을 때만) | — |

조정 로직은 `src/shllm/generate.py` 의 것을 그대로 쓴다 (8장). 원본 확률(`raw_prob`)은 조정 전 softmax 라 "조정이 무엇을 바꿨나"를 화면에서 비교할 수 있다.

## 사용 방법

```bash
docker compose up --build -d            # http://localhost:8082 대시보드 · http://localhost:8888 JupyterLab
docker compose logs -f dashboard lab
docker compose down
```

- `dashboard` 는 D 드라이브를 읽기 전용, `lab` 은 읽기·쓰기(노트북 02·07 이 토크나이저·체크포인트를 쓴다)로 마운트한다.
- `lab` 은 `src/`·`notebooks/`·`docs/`·`configs/`·`build/` 를 마운트하므로 코드 수정이 바로 반영된다 (`PYTHONPATH=/app/src`). 토큰·비밀번호 없음 — 로컬 전용.
- 노트북 탭의 "실행 결과"는 `bash scripts/verify.sh` 가 `build/notebooks/` 에 남긴 실행본이다. JupyterLab 에서 직접 실행한 결과는 그 노트북 파일 자체에 남는다(커밋 시 nbstripout 이 제거).

개발:

```bash
uv run uvicorn dashboard.api.main:app --port 8082 --reload   # API
cd dashboard/web && npm install && npm run dev               # http://localhost:5173, /api 는 8082 로 프록시
cd dashboard/web && npm run build                            # dist/ → FastAPI 가 서빙
```

체크포인트가 하나도 없으면 어텐션·생성 화면의 run 선택이 비어 있다 — `scripts/train.py` 를 먼저 돌린다.

## 검증

- `tests/test_dashboard_api.py` — 임시 run·체크포인트로 4개 엔드포인트 (verify.sh 에 포함)
- Playwright(MCP)로 1360×860 · 400×800 두 뷰포트에서 세 화면 확인 (2026-09-16, `.playwright-mcp/shot-*.png`)

## 관련 코드

- `dashboard/api/main.py` — 엔드포인트 전부
- `dashboard/web/src/App.tsx` (셸·해시 라우팅) · `views/{Runs,Attention,Generate,Book,Notebooks}.tsx` · `components/Heatmap.tsx` · `api.ts` (타입·호출) · `index.css` (DESIGN.md 토큰, `.prose-dark` 교재 스타일)
- `dashboard/Dockerfile` (target `dashboard` / `lab`, Noto CJK 폰트 포함) · `docker-compose.yml`
- `DESIGN.md` — 토큰·컴포넌트 계약. 값은 여기서만 정한다
