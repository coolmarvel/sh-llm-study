"""코퍼스 읽기.

data/corpus/korean-classics.txt   합본 (학습에 쓰는 기본 코퍼스)
data/corpus/works/*.txt           작품별 원문 (분석·비교용)
"""

from __future__ import annotations

from pathlib import Path

from shllm.config import CORPUS_DIR


def list_corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """합본 파일 목록 (works/ 하위는 포함하지 않는다 — 합본과 중복이므로)."""
    return sorted(corpus_dir.glob("*.txt"))


def list_work_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """작품별 파일 목록."""
    return sorted((corpus_dir / "works").glob("*.txt"))


def load_corpus(name: str | None = None, corpus_dir: Path = CORPUS_DIR) -> str:
    """name 이 없으면 corpus/ 의 모든 .txt 를 이어 붙여 돌려준다."""
    files = [corpus_dir / f"{name}.txt"] if name else list_corpus_files(corpus_dir)
    if not files:
        raise FileNotFoundError(
            f"{corpus_dir} 에 코퍼스가 없습니다. `uv run python scripts/download_corpus.py` 를 먼저 실행하세요."
        )
    return "\n\n".join(p.read_text("utf-8") for p in files)
