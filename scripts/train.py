"""긴 학습 (7장) — 노트북 밖에서 몇 시간 돌리는 스크립트.

    uv run python scripts/train.py --config configs/small-cpu.yaml            # 처음부터
    uv run python scripts/train.py --config configs/small-cpu.yaml --resume   # data/checkpoints/<run>/ckpt.pt 에서 이어서
    uv run python scripts/train.py --config configs/small-cpu.yaml --max-steps 100 --run-name smoke   # 설정 덮어쓰기

로그: data/runs/<run>/log.jsonl (대시보드가 읽는다) · 체크포인트: data/checkpoints/<run>/{ckpt,best}.pt
백그라운드 실행: nohup uv run python scripts/train.py --config ... > data/runs/<run>.out 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shllm.config import TOKENIZER_DIR, ensure_data_dirs, setup_cpu  # noqa: E402
from shllm.data import load_corpus  # noqa: E402
from shllm.model import GPT  # noqa: E402
from shllm.tokenizer import BPETokenizer  # noqa: E402
from shllm.train import Trainer, load_yaml_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="configs/*.yaml")
    ap.add_argument("--resume", action="store_true", help="같은 run 의 ckpt.pt 에서 이어서")
    ap.add_argument("--run-name", help="train.run_name 덮어쓰기")
    ap.add_argument("--max-steps", type=int, help="train.max_steps 덮어쓰기")
    ap.add_argument(
        "--threads", type=int, default=8, help="CPU 스레드 수 (작은 모델은 8 이 16 보다 빠르다)"
    )
    args = ap.parse_args()

    setup_cpu(num_threads=args.threads)
    ensure_data_dirs()
    model_cfg, train_cfg = load_yaml_config(args.config)
    if args.run_name:
        train_cfg.run_name = args.run_name
    if args.max_steps:
        train_cfg.max_steps = args.max_steps

    tok = BPETokenizer.load(TOKENIZER_DIR / train_cfg.tokenizer)
    if tok.vocab_size != model_cfg.vocab_size:
        raise SystemExit(
            f"토크나이저 어휘 {tok.vocab_size} ≠ model.vocab_size {model_cfg.vocab_size}"
        )
    t0 = time.time()
    data = torch.tensor(tok.encode(load_corpus("korean-classics")))  # D 드라이브는 한 번만 읽는다
    n = int(0.9 * len(data))
    print(
        f"코퍼스 {len(data):,} 토큰 (인코딩 {time.time() - t0:.0f}s) → train {n:,} / val {len(data) - n:,}"
    )

    model = GPT(model_cfg)
    print(f"모델 {model.n_params():,} 파라미터 (위치 임베딩 제외) | {model_cfg}")
    trainer = Trainer(model, data[:n], data[n:], train_cfg)
    if args.resume and trainer.resume():
        print(f"step {trainer.step} 에서 이어서")
    print(
        f"학습 {train_cfg.max_steps} 스텝, 스텝당 {train_cfg.batch_size * model_cfg.block_size:,} 토큰 → run '{train_cfg.run_name}'"
    )
    trainer.train()
    print(f"완료: best val {trainer.best_val:.3f}  ({(time.time() - t0) / 60:.0f}분)")


if __name__ == "__main__":
    main()
