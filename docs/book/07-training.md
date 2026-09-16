---
title: 7장 — 학습 루프: 이 PC 에서 몇 시간 안에
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 7장 — 학습 루프: 이 PC 에서 몇 시간 안에

6장까지로 모델은 완성됐다. 이 장의 질문은 **이 모델을 GPU 없는 PC 에서 몇 시간 안에 제대로 학습시키려면 무엇이 더
필요한가**다. 4장의 `train_steps` (배치 → forward → backward → step) 에 여섯 가지를 붙인다: 학습률 스케줄, AdamW 의
weight decay, 기울기 클리핑, 주기적 평가, JSONL 로그, 체크포인트. 노트북 `notebooks/07-training.ipynb`, 코드
`src/shllm/train.py`, 스크립트 `scripts/train.py`, 설정 `configs/*.yaml`.

## 7.1 Trainer 한 스텝

```python
lr = lr_at(self.step, cfg)                       # (1) 스케줄
for group in self.optimizer.param_groups: group["lr"] = lr
if self.step % cfg.eval_every == 0: ...          # (2) 평가·로그·best.pt
x, y = get_batch(train_data, block_size, batch_size, gen)
_, loss = self.model(x, y)                        # (3) 한 스텝
self.optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
self.optimizer.step()
if self.step % cfg.ckpt_every == 0: self.save_checkpoint()   # (4)
```

| 요소 | 무엇 | 왜 |
|---|---|---|
| warmup → cosine (`lr_at`) | 처음 100 스텝은 선형으로 올리고, 이후 코사인으로 `min_lr` 까지 줄인다 | 랜덤 초기 상태에서 큰 걸음을 밟으면 망가진다. 끝에는 세밀하게 |
| AdamW, β=(0.9, 0.95) | 파라미터마다 학습률을 보정하는 Adam + 분리된 weight decay | GPT-2/nanoGPT 관례. decay 는 행렬에만, bias·LayerNorm 은 제외 |
| `clip_grad_norm_(1.0)` | 기울기 벡터 전체 길이를 1 로 제한 | 가끔 튀는 배치가 학습을 뒤엎지 않게 |
| `estimate_loss` | train/val 각각 여러 배치 평균, `model.eval()` 로 dropout 끄고 | 한 배치의 loss 는 잡음이 크다 |
| `log.jsonl` | 평가 시점마다 한 줄 `{"step","train_loss","val_loss","lr","elapsed"}` | 사람·노트북·대시보드가 같은 파일을 읽는다 |
| `ckpt.pt` / `best.pt` | 최신 / val 최저 시점. 모델·옵티마이저·스텝·설정·토크나이저 이름 | 중단 후 `--resume`, 8장 생성은 `best.pt` |

## 7.2 파일 규약

```
data/runs/<run>/log.jsonl       평가 로그 (JSONL)
data/runs/<run>/config.yaml     모델·학습 설정 사본
data/checkpoints/<run>/ckpt.pt  최신 체크포인트
data/checkpoints/<run>/best.pt  val 최저
```

`load_checkpoint(run)` 이 `GPTConfig` 를 복원해 같은 모델을 만든다. 체크포인트에 **토크나이저 파일 이름**을 같이 넣는 이유:
토크나이저가 바뀌면(2장) 체크포인트는 쓸모없어지므로 어떤 것으로 학습했는지 기록해 둔다.

## 7.3 CPU 에서 빠르게

**스레드 수.** 4장에서 측정한 함정: 하이퍼스레드까지 16 스레드로 돌리면 8 스레드보다 16배 느렸다. `setup_cpu()` 기본은
물리 코어 수(8)다. 스레드가 물리 코어보다 많으면 캐시·동기화 경쟁이 계산보다 커진다.

**스텝당 토큰 수 = batch_size × block_size.** 이것이 예산이다. `small-cpu` 는 32 × 128 = 4,096 토큰/스텝, 스텝당 약 1초
(8 스레드, 6.8M 파라미터). 2,500 스텝 = 45분, 1,000만 토큰 (평가·체크포인트 포함).

**D 드라이브는 한 번만 읽는다.** `scripts/train.py` 는 시작 시 코퍼스를 통째로 인코딩해 텐서로 메모리에 둔다.
학습 루프 안에서 `/mnt/d` 를 반복 읽지 않는다 (CLAUDE.md 함정).

## 7.4 과적합 — 코퍼스가 작다는 것

코퍼스는 49만 BPE 토큰이다. 2,500 스텝이면 약 20 에폭, 6.8M 파라미터 모델은 문장을 외우기 시작한다. `small-cpu` 의
실제 결과 (2026-09-16, 8 스레드, 45분):

| step | train | val |
|---|---|---|
| 0 | 9.08 | 9.09 |
| 500 | 5.99 | 6.13 |
| 1000 | 5.40 | 5.76 |
| 2100 (best) | 4.77 | **5.485** |
| 2500 | 4.68 | 5.49 |

train 은 계속 내려가고 val 은 2,100 스텝에서 멈춘다 — 1장 n-gram, 4장 MLP 와 같은 현상. 학습률이 코사인으로 거의
0 에 가까워진 덕에 val 이 크게 튀어 오르진 않았지만, 더 돌려도 좋아질 여지는 없다. perplexity 로는 exp(5.485) ≈ 241 —
8,192 개 중 240 개로 좁힌 셈이고, 글자당으로는 약 3.8 bit (9장).

버티는 수단은 `dropout: 0.2`, `weight_decay: 0.1`, 그리고 val 최저 시점의 `best.pt` 다. 근본 해법은 **데이터**다.
nanoGPT 의 Tiny Shakespeare 도 같은 크기(110만 자)에서 같은 곡선을 그린다. 이 프로젝트의 다음 단계는 코퍼스 2차 확장
(한국어 위키 일부)이며, 이 결정은 `docs/todo.md` 에 기록한다.

## 7.5 실습 — 노트북 07

`notebooks/07-training.ipynb` (전체 실행 약 4분):

1. `configs/tiny-notebook.yaml` 로 600 스텝 학습 — 로그·설정·체크포인트 파일 확인, 학습률 곡선
2. `read_log("small-cpu")` 로 긴 학습의 train/val 곡선과 best 지점
3. `best.pt` 에서 생성
4. 학습된 토큰 임베딩의 이웃 (3장 다시 보기)

긴 학습은 노트북 밖에서:

```bash
uv run python scripts/train.py --config configs/small-cpu.yaml            # 처음부터
uv run python scripts/train.py --config configs/small-cpu.yaml --resume   # 이어서
nohup uv run python -u scripts/train.py --config configs/small-cpu.yaml > data/runs/small-cpu.out 2>&1 &
```

## 관련 코드

- `src/shllm/train.py` — `TrainConfig` · `lr_at` · `Trainer` (`estimate_loss` · `save_checkpoint` · `resume` · `train`) · `load_checkpoint` · `load_yaml_config` · `read_log`
- `scripts/train.py` — 설정 파일로 긴 학습, `--resume` `--run-name` `--max-steps` `--threads`
- `configs/tiny-notebook.yaml` (노트북용 600 스텝) · `configs/small-cpu.yaml` (v1.0 본 학습)
- `tests/test_train.py` — 스케줄 모양 · 학습이 loss 를 줄이고 로그·체크포인트를 남김 · 되살리기·이어 학습 · YAML 파싱
