"""학습 루프 (7장) — 4장 train_steps 에 실전 요소를 붙인 완성판.

    TrainConfig     학습 하이퍼파라미터 (배치·스텝·학습률 스케줄·평가 주기·체크포인트 주기)
    Trainer         한 run 을 책임진다: 데이터 → 배치 → 순전파/역전파 → AdamW → 로그(JSONL) → 체크포인트
    load_checkpoint 저장된 run 에서 모델(+설정)을 되살린다 — 8장 생성·9장 평가·대시보드가 쓴다

파일 규약 (config.RUNS_DIR / CHECKPOINT_DIR 아래 run 이름으로):
    data/runs/<run>/log.jsonl         한 줄 = 한 평가 시점 {"step", "train_loss", "val_loss", "lr", "elapsed"}
    data/runs/<run>/config.yaml       모델·학습 설정 사본
    data/checkpoints/<run>/ckpt.pt    최신 체크포인트 {"model", "optimizer", "step", "model_cfg", "train_cfg", "tokenizer"}
    data/checkpoints/<run>/best.pt    val loss 최저 시점

CPU 팁: 배치를 크게 잡기보다 block_size × batch_size 의 곱(스텝당 토큰 수)을 예산으로 생각한다.
7.6GB RAM 에서 V=8192, C=256 이면 스텝당 8~16K 토큰이 안전하다.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import yaml

from shllm.config import CHECKPOINT_DIR, RUNS_DIR
from shllm.data import get_batch
from shllm.model import GPT, GPTConfig


@dataclass
class TrainConfig:
    run_name: str = "run"
    batch_size: int = 32
    max_steps: int = 1000
    lr: float = 6e-4  # 최고 학습률
    min_lr: float = 6e-5  # 코사인 감쇠가 끝나는 값
    warmup_steps: int = 100  # 처음엔 작게 시작해 여기까지 선형으로 올린다
    weight_decay: float = 0.1
    grad_clip: float = 1.0  # 기울기 크기 상한 — 가끔 튀는 배치가 학습을 망치지 않게
    eval_every: int = 100
    eval_batches: int = 20  # 평가 때 train/val 각각 몇 배치 평균
    ckpt_every: int = 500
    seed: int = 1337
    tokenizer: str = "bpe-8192.json"  # data/tokenizers/ 안의 파일 이름 (체크포인트에 같이 기록)
    extra: dict = field(default_factory=dict)  # yaml 에 있는 그 밖의 키


def lr_at(step: int, cfg: TrainConfig) -> float:
    """warmup(선형 증가) → cosine(부드럽게 감소) → min_lr. GPT-2/nanoGPT 관례."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


class Trainer:
    def __init__(
        self,
        model: GPT,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        cfg: TrainConfig,
        runs_dir: Path = RUNS_DIR,
        ckpt_dir: Path = CHECKPOINT_DIR,
    ) -> None:
        self.model, self.cfg = model, cfg
        self.train_data, self.val_data = train_data, val_data
        self.run_dir = runs_dir / cfg.run_name
        self.ckpt_dir = ckpt_dir / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = self._make_optimizer()
        self.step = 0
        self.best_val = float("inf")
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.history: list[dict] = []

    def _make_optimizer(self) -> torch.optim.AdamW:
        # weight decay 는 행렬(2차원 이상)에만. bias·LayerNorm 은 빼는 것이 관례
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
        )

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """train/val 각각 eval_batches 배치의 평균 loss. 한 배치는 잡음이 커서 여러 개를 평균한다."""
        self.model.eval()  # dropout 끄기
        out = {}
        for name, data in (("train", self.train_data), ("val", self.val_data)):
            losses = []
            for _ in range(self.cfg.eval_batches):
                x, y = get_batch(data, self.model.cfg.block_size, self.cfg.batch_size, self.gen)
                _, loss = self.model(x, y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        self.model.train()
        return out

    def save_checkpoint(self, name: str = "ckpt.pt") -> Path:
        path = self.ckpt_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "best_val": self.best_val,
                "model_cfg": asdict(self.model.cfg),
                "train_cfg": asdict(self.cfg),
                "tokenizer": self.cfg.tokenizer,
            },
            path,
        )
        return path

    def resume(self, name: str = "ckpt.pt") -> bool:
        """체크포인트가 있으면 모델·옵티마이저·스텝을 되살린다. 없으면 False."""
        path = self.ckpt_dir / name
        if not path.exists():
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.step = ck["step"]
        self.best_val = ck.get("best_val", float("inf"))
        return True

    def _log(self, record: dict) -> None:
        self.history.append(record)
        with (self.run_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def train(self, verbose: bool = True) -> list[dict]:
        cfg = self.cfg
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {"model": asdict(self.model.cfg), "train": asdict(cfg)}, allow_unicode=True
            ),
            "utf-8",
        )
        self.model.train()
        t0 = time.time()
        while self.step < cfg.max_steps:
            # (1) 학습률 스케줄
            lr = lr_at(self.step, cfg)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            # (2) 평가·로그 — 스텝 0 도 기록해 곡선이 처음부터 보이게
            if self.step % cfg.eval_every == 0:
                losses = self.estimate_loss()
                rec = {
                    "step": self.step,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["val"], 4),
                    "lr": lr,
                    "elapsed": round(time.time() - t0, 1),
                }
                self._log(rec)
                if verbose:
                    print(
                        f"step {self.step:>6}  train {losses['train']:.3f}  val {losses['val']:.3f}  lr {lr:.2e}  {rec['elapsed']:.0f}s"
                    )
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.save_checkpoint("best.pt")
            # (3) 한 스텝
            x, y = get_batch(self.train_data, self.model.cfg.block_size, cfg.batch_size, self.gen)
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.step += 1
            # (4) 체크포인트
            if self.step % cfg.ckpt_every == 0 or self.step == cfg.max_steps:
                self.save_checkpoint()
        # 마지막 평가
        losses = self.estimate_loss()
        self._log(
            {
                "step": self.step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "lr": lr_at(self.step, cfg),
                "elapsed": round(time.time() - t0, 1),
            }
        )
        if losses["val"] < self.best_val:
            self.best_val = losses["val"]
            self.save_checkpoint("best.pt")
        self.save_checkpoint()
        return self.history


def load_checkpoint(
    run_name: str, name: str = "best.pt", ckpt_dir: Path = CHECKPOINT_DIR
) -> tuple[GPT, dict]:
    """run 의 체크포인트에서 GPT 를 만들어 돌려준다 (eval 모드). 두 번째 값은 체크포인트 dict (step, tokenizer 등)."""
    ck = torch.load(ckpt_dir / run_name / name, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def load_yaml_config(path: str | Path) -> tuple[GPTConfig, TrainConfig]:
    """configs/*.yaml → (GPTConfig, TrainConfig). 키는 model: / train: 두 블록."""
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    model_cfg = GPTConfig(**raw.get("model", {}))
    train_raw = dict(raw.get("train", {}))
    known = {f for f in TrainConfig.__dataclass_fields__ if f != "extra"}
    extra = {k: train_raw.pop(k) for k in list(train_raw) if k not in known}
    train_cfg = TrainConfig(**train_raw, extra=extra)
    return model_cfg, train_cfg


def read_log(run_name: str, runs_dir: Path = RUNS_DIR) -> list[dict]:
    path = runs_dir / run_name / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
