"""대시보드 API (M6), 학습 로그·체크포인트를 읽어 손실 곡선·어텐션 맵·토큰별 생성 확률을 제공한다.

    uv run uvicorn dashboard.api.main:app --port 8082 --reload      # 개발 (웹은 vite dev 서버가 /api 를 프록시)
    docker compose up                                              # 운영: 빌드된 웹 + API 를 8082 하나로

읽는 파일 (7장 규약): data/runs/<run>/{log.jsonl,config.yaml}, data/checkpoints/<run>/best.pt
모델은 run 마다 한 번 로드해 메모리에 둔다 (D 드라이브를 반복 읽지 않는다).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import markdown
import nbformat
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from nbconvert import HTMLExporter
from pydantic import BaseModel, Field
from pygments.formatters import HtmlFormatter

from shllm.config import CHECKPOINT_DIR, REPO_ROOT, RUNS_DIR, TOKENIZER_DIR, setup_cpu
from shllm.generate import adjust_logits, next_token_distribution
from shllm.tokenizer import BPETokenizer
from shllm.train import load_checkpoint, read_log

from .kernels import KernelPool

setup_cpu()
app = FastAPI(title="sh-llm-study dashboard", version="0.1")
WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
BOOK_DIR = REPO_ROOT / "docs" / "book"
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
EXECUTED_DIR = REPO_ROOT / "build" / "notebooks"  # verify.sh 가 실행한 노트북 (출력 포함)
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
KERNELS = KernelPool(cwd=NOTEBOOK_DIR, pythonpath=REPO_ROOT / "src")


# ---------- 실험 목록 · 손실 곡선 ----------


def _run_names() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "log.jsonl").exists())


def _run_summary(name: str) -> dict:
    log = read_log(name)
    cfg_path = RUNS_DIR / name / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text("utf-8")) if cfg_path.exists() else {}
    best = min(log, key=lambda r: r["val_loss"]) if log else None
    m = cfg.get("model", {})
    n_params = None
    if m:
        C, L, V = m.get("n_embd", 0), m.get("n_layer", 0), m.get("vocab_size", 0)
        n_params = V * C + L * (12 * C * C + 13 * C) + 2 * C  # 6장 식 (bias·LayerNorm 포함 근사)
    return {
        "name": name,
        "model": m,
        "train": cfg.get("train", {}),
        "n_params": n_params,
        "steps": log[-1]["step"] if log else 0,
        "elapsed": log[-1]["elapsed"] if log else 0,
        "best_val": best["val_loss"] if best else None,
        "best_step": best["step"] if best else None,
        "has_checkpoint": (CHECKPOINT_DIR / name / "best.pt").exists(),
    }


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return [_run_summary(n) for n in _run_names()]


@app.get("/api/runs/{name}/log")
def run_log(name: str) -> list[dict]:
    if name not in _run_names():
        raise HTTPException(404, f"run 없음: {name}")
    return read_log(name)


# ---------- 모델 로드 ----------


@lru_cache(maxsize=4)
def _load(name: str):
    try:
        model, ck = load_checkpoint(name, "best.pt")
    except FileNotFoundError as e:
        raise HTTPException(404, f"체크포인트 없음: {name}") from e
    tok = BPETokenizer.load(TOKENIZER_DIR / ck.get("tokenizer", "bpe-8192.json"))
    return model, tok, ck


@app.get("/api/checkpoints")
def list_checkpoints() -> list[dict]:
    if not CHECKPOINT_DIR.exists():
        return []
    return [
        {"name": p.name, "files": sorted(f.name for f in p.glob("*.pt"))}
        for p in sorted(CHECKPOINT_DIR.iterdir())
        if (p / "best.pt").exists()
    ]


# ---------- 어텐션 맵 ----------


class AttentionRequest(BaseModel):
    run: str
    text: str = Field(min_length=1, max_length=400)


@app.post("/api/attention")
def attention(req: AttentionRequest) -> dict:
    model, tok, _ = _load(req.run)
    ids = tok.encode(req.text)[: model.cfg.block_size]
    with torch.no_grad():
        model(torch.tensor([ids]))
    maps = model.attention_maps()  # 층별 (1, H, T, T)
    return {
        "tokens": [tok.token_str(i) for i in ids],
        "n_layer": len(maps),
        "n_head": maps[0].shape[1],
        "maps": [
            [m[0, h].tolist() for h in range(m.shape[1])] for m in maps
        ],  # [layer][head][T][T]
    }


# ---------- 생성 (토큰별 확률) ----------


class GenerateRequest(BaseModel):
    run: str
    prompt: str = Field(min_length=1, max_length=400)
    max_new_tokens: int = Field(60, ge=1, le=300)
    temperature: float = Field(0.8, ge=0.0, le=3.0)
    top_k: int | None = Field(50, ge=1, le=8192)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=3.0)
    top_n: int = Field(10, ge=1, le=50)  # 자리마다 돌려줄 후보 수
    seed: int = 0


@app.post("/api/generate")
def generate_api(req: GenerateRequest) -> dict:
    """한 토큰씩 뽑으면서 자리마다 (뽑힌 토큰, 그 확률, 상위 후보 top_n) 을 기록한다."""
    model, tok, _ = _load(req.run)
    idx = torch.tensor([tok.encode(req.prompt)])
    g = torch.Generator().manual_seed(req.seed)
    steps = []
    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            # 조정 전 원본 분포도 같이 (사용자가 "조정이 무엇을 바꿨나" 를 보게)
            block = model.cfg.block_size
            logits, _ = model(idx[:, -block:])
            raw = torch.softmax(logits[:, -1, :], dim=-1)[0]
            probs = next_token_distribution(
                model, idx, req.temperature, req.top_k, req.top_p, req.repetition_penalty
            )[0]
            nxt = int(torch.multinomial(probs, 1, generator=g))
            top = torch.topk(probs, req.top_n)
            steps.append(
                {
                    "token": tok.token_str(nxt),
                    "id": nxt,
                    "prob": float(probs[nxt]),
                    "raw_prob": float(raw[nxt]),
                    "candidates": [
                        {
                            "token": tok.token_str(int(i)),
                            "prob": float(p),
                            "raw_prob": float(raw[int(i)]),
                        }
                        for p, i in zip(top.values, top.indices, strict=False)
                    ],
                }
            )
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    return {
        "prompt_tokens": [tok.token_str(i) for i in idx[0, : -len(steps)].tolist()],
        "steps": steps,
        "text": tok.decode(idx[0].tolist()),
    }


def _generate_steps(req: GenerateRequest):
    """generate_api 와 같은 계산을 한 스텝씩 yield, 스트리밍용."""
    model, tok, _ = _load(req.run)
    idx = torch.tensor([tok.encode(req.prompt)])
    g = torch.Generator().manual_seed(req.seed)
    yield {"prompt_tokens": [tok.token_str(i) for i in idx[0].tolist()]}
    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            block = model.cfg.block_size
            logits, _ = model(idx[:, -block:])
            raw = torch.softmax(logits[:, -1, :], dim=-1)[0]
            probs = next_token_distribution(
                model, idx, req.temperature, req.top_k, req.top_p, req.repetition_penalty
            )[0]
            nxt = int(torch.multinomial(probs, 1, generator=g))
            top = torch.topk(probs, req.top_n)
            yield {
                "token": tok.token_str(nxt),
                "id": nxt,
                "prob": float(probs[nxt]),
                "raw_prob": float(raw[nxt]),
                "candidates": [
                    {
                        "token": tok.token_str(int(i)),
                        "prob": float(p),
                        "raw_prob": float(raw[int(i)]),
                    }
                    for p, i in zip(top.values, top.indices, strict=False)
                ],
            }
            idx = torch.cat([idx, torch.tensor([[nxt]])], dim=1)
    yield {"text": tok.decode(idx[0].tolist())}


@app.post("/api/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
    """NDJSON 스트림: 첫 줄 {prompt_tokens}, 이후 한 줄에 토큰 하나, 마지막 줄 {text}."""

    def lines():
        for item in _generate_steps(req):
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/api/tokenize")
def tokenize(run: str, text: str) -> dict:
    _, tok, _ = _load(run)
    ids = tok.encode(text)
    return {"ids": ids, "tokens": [tok.token_str(i) for i in ids]}


# ---------- 교재 · 노트북 ----------


def _chapters() -> list[Path]:
    return sorted(BOOK_DIR.glob("[0-9][0-9]-*.md")) if BOOK_DIR.exists() else []


def _title(path: Path) -> str:
    m = FRONTMATTER.match(path.read_text("utf-8"))
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip()
    return path.stem


@app.get("/api/book")
def list_book() -> list[dict]:
    return [{"name": p.stem, "title": _title(p)} for p in _chapters()]


@app.get("/api/book/{name}")
def book_chapter(name: str) -> dict:
    path = BOOK_DIR / f"{name}.md"
    if not path.is_file() or path not in _chapters():
        raise HTTPException(404, f"챕터 없음: {name}")
    text = FRONTMATTER.sub("", path.read_text("utf-8"), count=1)
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
        extension_configs={"codehilite": {"guess_lang": False}},
    )
    return {
        "name": name,
        "title": _title(path),
        "html": html,
        "css": HtmlFormatter(style="monokai").get_style_defs(".codehilite"),
    }


@app.get("/api/notebooks")
def list_notebooks() -> list[dict]:
    out = []
    for p in sorted(NOTEBOOK_DIR.glob("[0-9][0-9]-*.ipynb")) if NOTEBOOK_DIR.exists() else []:
        executed = EXECUTED_DIR / p.name
        nb = nbformat.read(p, as_version=4)
        first = next((c.source for c in nb.cells if c.cell_type == "markdown"), "")
        title = first.splitlines()[0].lstrip("# ").strip() if first else p.stem
        out.append(
            {
                "name": p.stem,
                "title": title,
                "cells": len(nb.cells),
                "executed": executed.exists(),
                "executed_at": executed.stat().st_mtime if executed.exists() else None,
                "lab_url": f"http://localhost:8888/lab/tree/notebooks/{p.name}",
            }
        )
    return out


@lru_cache(maxsize=16)
def _notebook_html(path: str, mtime: float) -> str:
    nb = nbformat.read(path, as_version=4)
    exporter = HTMLExporter(
        template_name="lab", exclude_input_prompt=True, exclude_output_prompt=True
    )
    body, _ = exporter.from_notebook_node(nb)
    return body


@app.get("/api/notebooks/{name}/html", response_class=HTMLResponse)
def notebook_html(name: str, executed: bool = True) -> str:
    """실행된 노트북(build/notebooks, 출력 포함) 을 HTML 로. executed=false 면 원본(출력 없음)."""
    path = (EXECUTED_DIR if executed else NOTEBOOK_DIR) / f"{name}.ipynb"
    if not path.is_file():
        raise HTTPException(
            404, f"노트북 없음: {path.name} (실행본은 bash scripts/verify.sh 가 만든다)"
        )
    return _notebook_html(str(path), path.stat().st_mtime)


# ---------- 노트북 셀 편집 · 실행 (웹 내장 실행기) ----------


class CellsPayload(BaseModel):
    cells: list[dict]  # [{"id","cell_type","source"}]


@app.get("/api/notebooks/{name}/cells")
def notebook_cells(name: str) -> dict:
    path = NOTEBOOK_DIR / f"{name}.ipynb"
    if not path.is_file():
        raise HTTPException(404, f"노트북 없음: {name}")
    nb = nbformat.read(path, as_version=4)
    # 실행본이 있으면 그 출력을 초기값으로 (verify.sh 결과), 실행 전에도 결과를 볼 수 있게
    executed = EXECUTED_DIR / f"{name}.ipynb"
    outputs: dict[str, list] = {}
    if executed.exists():
        ex = nbformat.read(executed, as_version=4)
        if len(ex.cells) == len(nb.cells):
            for a, b in zip(nb.cells, ex.cells, strict=False):
                if b.cell_type == "code":
                    outputs[a.get("id", "")] = [dict(o) for o in b.get("outputs", [])]
    return {
        "name": name,
        "cells": [
            {
                "id": c.get("id", str(i)),
                "cell_type": c.cell_type,
                "source": c.source,
                "outputs": outputs.get(c.get("id", ""), []),
            }
            for i, c in enumerate(nb.cells)
        ],
        "kernel_alive": name in KERNELS.names(),
    }


@app.put("/api/notebooks/{name}/cells")
def save_notebook_cells(name: str, payload: CellsPayload) -> dict:
    """셀 소스를 노트북 파일에 저장 (출력은 저장하지 않는다, git 규약과 같게). 순번 id 로 정규화."""
    path = NOTEBOOK_DIR / f"{name}.ipynb"
    if not path.is_file():
        raise HTTPException(404, f"노트북 없음: {name}")
    nb = nbformat.read(path, as_version=4)
    nb.cells = [
        nbformat.v4.new_markdown_cell(c["source"])
        if c["cell_type"] == "markdown"
        else nbformat.v4.new_code_cell(c["source"])
        for c in payload.cells
    ]
    for i, c in enumerate(nb.cells):
        c.id = str(i)
        c.metadata = nbformat.NotebookNode()
    nbformat.write(nb, path)
    return {"saved": len(nb.cells)}


class ExecRequest(BaseModel):
    code: str


@app.post("/api/notebooks/{name}/execute")
def execute_cell(name: str, req: ExecRequest) -> StreamingResponse:
    """NDJSON 스트림, 출력이 생길 때마다 한 줄. 커널은 노트북마다 하나, 처음 호출에 뜬다 (수 초)."""

    def lines():
        for out in KERNELS.execute(name, req.code):
            yield json.dumps(out, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/api/notebooks/{name}/kernel/{action}")
def kernel_action(name: str, action: str) -> dict:
    if action == "interrupt":
        KERNELS.interrupt(name)
    elif action == "restart":
        KERNELS.restart(name)
    elif action == "shutdown":
        KERNELS.shutdown(name)
    elif action == "start":
        KERNELS.get(name)
    else:
        raise HTTPException(400, f"알 수 없는 동작: {action}")
    return {"name": name, "alive": name in KERNELS.names()}


@app.get("/api/kernels")
def list_kernels() -> list[str]:
    return KERNELS.names()


@app.get("/api/source")
def source_file(path: str) -> dict:
    """저장소 안의 소스 파일을 하이라이트한 HTML 로, 교재의 '관련 코드' 링크용. src/ scripts/ configs/ dashboard/ tests/ 만."""
    rel = Path(path)
    if (
        rel.is_absolute()
        or ".." in rel.parts
        or rel.parts[0] not in ("src", "scripts", "configs", "tests", "dashboard", "docs")
    ):
        raise HTTPException(400, "허용되지 않는 경로")
    full = REPO_ROOT / rel
    if not full.is_file():
        raise HTTPException(404, f"파일 없음: {path}")
    from pygments import highlight
    from pygments.lexers import get_lexer_for_filename
    from pygments.util import ClassNotFound

    code = full.read_text("utf-8")
    try:
        lexer = get_lexer_for_filename(full.name)
    except ClassNotFound:
        from pygments.lexers import TextLexer

        lexer = TextLexer()
    html = highlight(code, lexer, HtmlFormatter(cssclass="codehilite", linenos="table"))
    return {
        "path": path,
        "lines": code.count("\n") + 1,
        "html": html,
        "css": HtmlFormatter(style="monokai").get_style_defs(".codehilite"),
    }


@app.on_event("shutdown")
def _shutdown_kernels() -> None:
    KERNELS.shutdown()


# ---------- 정적 웹 (빌드 결과) ----------

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        file = WEB_DIST / path
        return FileResponse(file if path and file.is_file() else WEB_DIST / "index.html")


_ = adjust_logits  # (참고) 조정 로직은 shllm.generate 에 있다. API 는 그것을 그대로 쓴다
