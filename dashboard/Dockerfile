# 이미지 두 개를 한 파일로:
#   dashboard  1단계 웹 빌드(node) → python(uv) + shllm + dist 를 FastAPI 가 서빙 (8082)
#   lab        JupyterLab (8888) — 노트북을 브라우저에서 실행. src/notebooks/docs 는 마운트해서 쓴다
FROM node:24-slim AS web
WORKDIR /web
COPY dashboard/web/package.json dashboard/web/package-lock.json ./
RUN npm ci
COPY dashboard/web ./
RUN npm run build

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONPATH=/app/src SHLLM_DATA_DIR=/data SHLLM_THREADS=4
# 한글 폰트 (matplotlib 그래프 라벨·PDF) — Noto CJK
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md LICENSE ./

FROM base AS dashboard
RUN uv sync --frozen --no-dev --group dashboard --no-install-project
COPY src ./src
COPY dashboard/__init__.py ./dashboard/
COPY dashboard/api ./dashboard/api
COPY --from=web /web/dist ./dashboard/web/dist
EXPOSE 8082
CMD ["uv", "run", "--no-sync", "uvicorn", "dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8082"]

FROM base AS lab
# dev 그룹 = jupyterlab · nbconvert · matplotlib 등. 프로젝트 코드는 마운트(/app/src) 로 최신을 쓴다
RUN uv sync --frozen --no-install-project
EXPOSE 8888
CMD ["uv", "run", "--no-sync", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.root_dir=/app", "--ServerApp.default_url=/lab/tree/notebooks"]
