"""경로·하드웨어 설정. 값의 소유권은 이 모듈에 있다, 다른 곳에서 경로를 하드코딩하지 않는다."""

from __future__ import annotations

import os
from pathlib import Path

import torch

# 저장소 루트 = src/shllm/config.py 의 두 단계 위
REPO_ROOT = Path(__file__).resolve().parents[2]

# data/ 는 /mnt/d/sh-llm-data 로 가는 심볼릭 링크 (ADR-0002). 환경변수로 덮어쓸 수 있다.
DATA_DIR = Path(os.environ.get("SHLLM_DATA_DIR", REPO_ROOT / "data"))
CORPUS_DIR = DATA_DIR / "corpus"
TOKENIZER_DIR = DATA_DIR / "tokenizers"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
RUNS_DIR = DATA_DIR / "runs"


def ensure_data_dirs() -> None:
    """데이터 디렉토리가 없으면 만든다 (심볼릭 링크가 끊긴 경우 명확한 에러를 낸다)."""
    if DATA_DIR.is_symlink() and not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} 심볼릭 링크의 대상이 없습니다. D 드라이브가 마운트됐는지 확인하세요."
        )
    for d in (CORPUS_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# 물리 코어 수. os.cpu_count() 는 하이퍼스레드까지 세서(16) 두 배가 나온다, 작은 모델은 16 스레드가 8 보다 16배 느렸다
# (2026-09-16 측정: MLP 스텝당 8 스레드 17ms, 16 스레드 275ms, 스레드끼리 캐시·동기화 경쟁). 환경변수로 덮어쓸 수 있다.
PHYSICAL_CORES = int(os.environ.get("SHLLM_THREADS", max(1, (os.cpu_count() or 2) // 2)))


def setup_cpu(num_threads: int | None = None, seed: int = 1337) -> torch.device:
    """CPU 전용 설정: 스레드 수(기본 = 물리 코어 수)와 시드를 고정하고 device 를 돌려준다."""
    torch.set_num_threads(num_threads or PHYSICAL_CORES)
    torch.manual_seed(seed)
    return torch.device("cpu")
