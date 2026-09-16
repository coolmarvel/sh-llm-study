"""BPE 토크나이저 학습 (2장), 코퍼스 합본 하나로 data/tokenizers/<이름>.json 을 만든다.

uv run python scripts/train_tokenizer.py --corpus korean-mixed --vocab-size 8192 --out bpe-8192-mixed.json
uv run python scripts/train_tokenizer.py --corpus korean-classics --out bpe-8192.json     # 2장 노트북과 같은 결과
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shllm.config import TOKENIZER_DIR, ensure_data_dirs  # noqa: E402
from shllm.data import load_corpus  # noqa: E402
from shllm.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", default="korean-classics", help="data/corpus/<이름>.txt")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--min-char-count", type=int, default=2)
    ap.add_argument("--out", required=True, help="data/tokenizers/ 안의 파일 이름")
    args = ap.parse_args()

    ensure_data_dirs()
    text = load_corpus(args.corpus)
    print(f"코퍼스 {args.corpus}: {len(text):,} 자")
    t0 = time.time()
    tok = BPETokenizer.train(
        text, args.vocab_size, min_char_count=args.min_char_count, verbose=True
    )
    ids = tok.encode(text)
    path = TOKENIZER_DIR / args.out
    tok.save(path)
    print(
        f"{path}: 어휘 {tok.vocab_size:,}, 코퍼스 {len(ids):,} 토큰 (토큰당 {len(text) / len(ids):.2f} 자), {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
