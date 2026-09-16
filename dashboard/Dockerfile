# 대시보드 이미지: 1단계 웹 빌드(node) → python(uv) + shllm + dist 를 FastAPI 가 서빙 (8082)
# 노트북 실행기는 같은 프로세스가 ipykernel 을 띄운다 (별도 JupyterLab 서버 없음). src/notebooks/docs/configs 는 마운트.
FROM node:24-slim AS web
WORKDIR /web
COPY dashboard/web/package.json dashboard/web/package-lock.json ./
RUN npm ci
COPY dashboard/web ./
RUN npm run build

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONPATH=/app/src SHLLM_DATA_DIR=/data SHLLM_THREADS=8
# 한글 폰트 (matplotlib 그래프 라벨·PDF). Noto CJK
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
