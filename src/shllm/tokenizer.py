"""토크나이저 — 텍스트를 정수 열로, 정수 열을 텍스트로.

1장: CharTokenizer (문자 하나 = 토큰 하나). 가장 단순하지만 어휘가 크고 시퀀스가 길다.
2장: BPETokenizer (자주 붙어 나오는 문자쌍을 병합) — M1 에서 추가.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """문자 단위 토크나이저. 코퍼스에 등장한 모든 문자에 정수 id 를 하나씩 준다."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        return cls(json.loads(Path(path).read_text("utf-8"))["chars"])
