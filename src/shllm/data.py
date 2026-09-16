"""코퍼스 읽기 + 작품 목록(SSOT).

data/corpus/korean-classics.txt   합본 (학습에 쓰는 기본 코퍼스)
data/corpus/works/*.txt           작품별 원문 (분석·비교용). 파일명 = WORKS 의 slug
"""

from __future__ import annotations

from pathlib import Path

import torch

from shllm.config import CORPUS_DIR

# (작가, 제목, 파일 slug). scripts/download_corpus.py 가 이 목록으로 받고, 10장 SFT 시연이 질문-답을 만든다.
WORKS: list[tuple[str, str, str]] = [
    ("김유정", "봄봄", "kim-yujeong-bombom"),
    ("김유정", "동백꽃", "kim-yujeong-dongbaekkkot"),
    ("김유정", "소낙비", "kim-yujeong-sonakbi"),
    ("김유정", "만무방", "kim-yujeong-manmubang"),
    ("현진건", "운수 좋은 날", "hyun-jingeon-unsu-joeun-nal"),
    ("현진건", "빈처", "hyun-jingeon-bincheo"),
    ("현진건", "B사감과 러브레터", "hyun-jingeon-b-sagam"),
    ("현진건", "술 권하는 사회", "hyun-jingeon-sul-gwonhaneun-sahoe"),
    ("이상", "날개", "yi-sang-nalgae"),
    ("이효석", "메밀꽃 필 무렵", "lee-hyoseok-memilkkot"),
    ("나도향", "벙어리 삼룡이", "na-dohyang-samryongi"),
    ("나도향", "물레방아", "na-dohyang-mullebanga"),
    ("김동인", "감자", "kim-dongin-gamja"),
    ("김동인", "배따라기", "kim-dongin-baettaragi"),
    ("김동인", "광염 소나타", "kim-dongin-gwangyeom-sonata"),
    ("채만식", "레디메이드 인생", "chae-manshik-readymade"),
    ("채만식", "치숙", "chae-manshik-chisuk"),
    ("채만식", "탁류", "chae-manshik-takryu"),
    ("채만식", "태평천하", "chae-manshik-taepyeongcheonha"),
    ("이광수", "무정", "lee-gwangsu-mujeong"),
]


def list_corpus_files(corpus_dir: Path = CORPUS_DIR) -> list[Path]:
    """합본 파일 목록 (works/ 하위는 포함하지 않는다, 합본과 중복이므로)."""
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


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """토큰 열에서 무작위 위치 batch_size 곳을 골라 (x, y) 를 만든다.

    x[i] = data[s : s+T],  y[i] = data[s+1 : s+T+1], y 는 x 를 한 칸 민 것. 자리마다 "다음 토큰" 이 정답.
    반환 shape: x (B, T), y (B, T). 4장 MLP 는 y 의 마지막 열만 쓰고, 6장 GPT 는 T 개 자리를 한 번에 학습한다.
    """
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y
