"""토크나이저, 텍스트를 정수 열로, 정수 열을 텍스트로.

1장: CharTokenizer (문자 하나 = 토큰 하나). 가장 단순하지만 어휘가 크고 시퀀스가 길다.
2장: BPETokenizer (바이트에서 출발해 자주 붙어 나오는 쌍을 병합). 어휘 크기를 마음대로 정하고,
     어떤 입력이든(처음 보는 글자·이모지) 토큰화할 수 있다.

두 클래스는 같은 인터페이스: vocab_size · encode(text) · decode(ids) · save(path) · load(path).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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


class BPETokenizer:
    """바이트 단위 BPE (Byte Pair Encoding). GPT-2 이후 대부분의 LLM 이 쓰는 방식.

    기본 어휘 = 바이트 256개 (0~255). 그 위에 **병합 규칙**을 순서대로 쌓는다:
        merges[i] = (a, b)  →  토큰 a 바로 뒤에 b 가 오면 새 토큰 256+i 로 합친다
    학습 = "지금 가장 자주 붙어 나오는 쌍"을 찾아 병합 규칙에 추가하기를 vocab_size 에 닿을 때까지 반복.
    인코딩 = 텍스트를 UTF-8 바이트로 바꾼 뒤 병합 규칙을 배운 순서대로 적용.
    디코딩 = 토큰마다 바이트열을 이어 붙여 UTF-8 로 되돌림.

    한글 한 글자는 UTF-8 로 3바이트라 처음 수백 번의 병합은 바이트를 음절로 되돌리는 데 쓰인다 (교재 2.3).
    텍스트는 먼저 PATTERN 으로 조각(단어·구두점·공백)을 낸 뒤 조각 안에서만 병합한다, 토큰이 단어 경계를 넘지 않게.
    """

    # " ?\w+" = 선택적 공백 + 글자/숫자 (\w 는 유니코드라 한글·한자 포함), " ?[^\s\w]+" = 구두점 묶음, "\s+" = 남은 공백
    PATTERN = r" ?\w+| ?[^\s\w]+|\s+"

    def __init__(self, merges: list[tuple[int, int]], pattern: str = PATTERN) -> None:
        self.merges = [tuple(m) for m in merges]
        self.pattern = pattern
        self._re = re.compile(pattern)
        # rank: 쌍 → 병합 순서. 인코딩 때 "가장 먼저 배운 쌍"부터 합치려고 쓴다
        self.rank: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        # vocab: 토큰 id → 바이트열. 병합 토큰은 두 부분의 바이트열을 이어 붙인 것
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}  # 조각 → ids. 같은 어절이 반복되므로 효과가 크다

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    # ---------- 학습 ----------

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        char_first: bool = True,
        min_char_count: int = 2,
        pattern: str = PATTERN,
        verbose: bool = False,
    ) -> BPETokenizer:
        """text 에서 병합 규칙을 배운다 (총 어휘 = 256 + 병합 수 = vocab_size).

        char_first=True 면 빈도 병합에 앞서 **글자 조립 병합**을 먼저 넣는다: text 에 min_char_count 번 이상
        나온 여러 바이트짜리 글자(한글 음절 = 3바이트)를 바이트에서 통째로 만드는 규칙. 이게 없으면 "옛" 의
        끝 바이트와 "날" 의 첫 바이트처럼 **글자 경계를 걸치는 조각**이 먼저 합쳐져 어휘가 깨진 바이트 토큰으로
        가득 찬다 (교재 2.3. GPT-2 토크나이저가 한글에 불리한 이유). 드문 글자는 여전히 바이트로 남는다.

        조각을 Counter 로 묶어 (고유 조각, 빈도) 로 다루면 같은 어절을 수만 번 다시 세지 않는다.
        또 쌍의 빈도와 "그 쌍이 들어 있는 조각 목록"을 유지해, 병합할 때 관련 조각만 고친다,
        병합마다 전체를 다시 세면 코퍼스 114만 자에서 수천 번 병합에 10분이 넘는다 (증분이면 1분).
        """
        if vocab_size < 256:
            raise ValueError("vocab_size 는 256 이상이어야 합니다 (바이트 256개가 기본 어휘)")
        chunks = Counter(re.findall(pattern, text))
        words: list[tuple[list[int], int]] = [
            (list(c.encode("utf-8")), n) for c, n in chunks.items()
        ]

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)  # 쌍 → 등장 빈도
        pair_words: dict[tuple[int, int], set[int]] = defaultdict(
            set
        )  # 쌍 → 그 쌍을 가진 조각 번호들
        for wi, (ids, n) in enumerate(words):
            for pair in zip(ids, ids[1:], strict=False):  # zip(ids, ids[1:]) = 인접 쌍 순회
                pair_counts[pair] += n
                pair_words[pair].add(wi)

        merges: list[tuple[int, int]] = []
        rank: dict[tuple[int, int], int] = {}

        def apply_merge(pair: tuple[int, int]) -> int:
            """pair 를 새 토큰으로 등록하고, 그 쌍이 들어 있는 조각들만 고쳐 쌍 빈도를 갱신한다."""
            new_id = 256 + len(merges)
            merges.append(pair)
            rank[pair] = new_id
            for wi in list(pair_words[pair]):
                ids, n = words[wi]
                for p in zip(ids, ids[1:], strict=False):  # 옛 쌍들을 빼고
                    pair_counts[p] -= n
                    pair_words[p].discard(wi)
                ids = _merge(ids, pair, new_id)
                words[wi] = (ids, n)
                for p in zip(ids, ids[1:], strict=False):  # 새 쌍들을 더한다
                    pair_counts[p] += n
                    pair_words[p].add(wi)
            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            return new_id

        if char_first:
            char_counts = Counter(text)
            for (
                ch,
                n,
            ) in char_counts.most_common():  # 흔한 글자부터, 어휘가 모자라면 드문 글자가 밀린다
                if n < min_char_count or len(merges) >= vocab_size - 256:
                    break
                ids = list(ch.encode("utf-8"))
                while (
                    len(ids) > 1 and len(merges) < vocab_size - 256
                ):  # 앞에서부터 한 바이트씩 붙인다
                    pair = (ids[0], ids[1])
                    tok = (
                        rank[pair] if pair in rank else apply_merge(pair)
                    )  # 앞부분이 같은 글자끼리 공유
                    ids = [tok] + ids[2:]
            if verbose:
                print(f"글자 조립 병합 {len(merges)}개 (min_char_count={min_char_count})")

        while len(merges) < vocab_size - 256:
            live = {p: c for p, c in pair_counts.items() if c > 0}
            if not live:
                break  # 더 합칠 쌍이 없다 (텍스트가 아주 짧을 때)
            best = max(live, key=live.get)  # 가장 빈번한 쌍
            if verbose and len(merges) % 1000 == 0:
                print(f"merge {len(merges)}: {best} ({live[best]}회)")
            apply_merge(best)
        return cls(merges, pattern)

    def truncated(self, vocab_size: int) -> BPETokenizer:
        """앞쪽 병합 규칙만 남긴 더 작은 토크나이저. 병합은 순서대로 쌓이므로 어휘 크기 실험을 다시 학습 없이 한다."""
        return BPETokenizer(self.merges[: vocab_size - 256], self.pattern)

    # ---------- 인코딩 / 디코딩 ----------

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # 지금 있는 인접 쌍 중 가장 먼저 배운(rank 가 낮은) 쌍을 찾아 합친다. 없으면 끝
            pairs = set(zip(ids, ids[1:], strict=False))
            pair = min(pairs, key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = _merge(ids, pair, 256 + self.rank[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        # errors="replace": 토큰 경계가 UTF-8 글자 중간에 걸리면(불완전한 바이트열) � 로 표시한다
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, token_id: int) -> str:
        """토큰 하나를 사람이 읽을 문자열로 (불완전한 바이트는 <0xNN> 로)."""
        try:
            return self.vocab[token_id].decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"<0x{b:02X}>" for b in self.vocab[token_id])

    # ---------- 저장 / 로드 ----------

    def save(self, path: str | Path) -> None:
        data = {"type": "bpe", "pattern": self.pattern, "merges": self.merges}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text("utf-8"))
        return cls([tuple(m) for m in data["merges"]], data["pattern"])


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """ids 안의 모든 pair 를 new_id 하나로 바꾼 새 리스트."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
