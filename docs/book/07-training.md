---
title: 7장 학습 루프: 이 PC 에서 몇 시간 안에
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 7장 학습 루프: 이 PC 에서 몇 시간 안에

6장까지로 모델은 완성됐다. 이 장의 질문은 **이 모델을 GPU 없는 PC 에서 몇 시간 안에 제대로 학습시키려면 무엇이 더
필요한가**다. 4장의 `train_steps` (배치 → forward → backward → step) 에 여섯 가지를 붙인다: 학습률 스케줄, AdamW 의
weight decay, 기울기 클리핑, 주기적 평가, JSONL 로그, 체크포인트. 노트북 `notebooks/07-training.ipynb`, 코드
`src/shllm/train.py`, 스크립트 `scripts/train.py`, 설정 `configs/*.yaml`.

## 읽기 전에

- 4장의 학습 루프 세 줄(`zero_grad · backward · step`), 6장의 `GPT.forward(idx, targets)`, 1장의 train/val 분리.
- **에폭(epoch)**: 학습 데이터를 한 바퀴 다 본 양. 코퍼스 49만 토큰, 스텝당 4,096 토큰이면 120 스텝이 1 에폭.
- YAML: 들여쓰기로 계층을 표현하는 설정 파일. `model:` 과 `train:` 두 블록이 각각 `GPTConfig`, `TrainConfig` 가 된다.

## 핵심 용어

| 용어 | 한 줄 정의 | 비유 |
|---|---|---|
| **스텝(step)** | 배치 하나로 파라미터를 한 번 갱신 | 반복문의 한 회전 |
| **warmup** | 처음 N 스텝 동안 학습률을 0 에서 선형으로 올리기 | 엔진 예열 |
| **cosine 감쇠** | 학습률을 코사인 곡선으로 부드럽게 줄이기 | 브레이크를 서서히 |
| **weight decay** | 파라미터를 매 스텝 조금씩 0 쪽으로 당기는 정규화 | 큰 값에 세금 |
| **grad clip** | 기울기 벡터의 길이가 상한을 넘으면 줄이기 | 과속 방지턱 |
| **체크포인트** | 모델·옵티마이저·스텝을 파일로 저장한 스냅샷 | 게임 세이브 |
| **best.pt / ckpt.pt** | val 최저 시점 / 가장 최근 | 최고 기록 / 자동 저장 |
| **run** | 설정 하나로 돌린 학습 한 번. 로그·체크포인트 폴더 이름 | 실험 ID |
| **과적합** | train 은 내려가는데 val 은 오르는 상태 | 기출문제만 외운 학생 |
| **드롭아웃 확률** | 학습 중 0 으로 만드는 활성값의 비율 | 결석률 |

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

### 단계별 예시: 스텝 하나가 처리하는 것

`small-cpu` 설정으로 스텝 하나를 따라간다.

1. `get_batch` 가 48만 토큰 열에서 무작위 32 곳을 골라 `x (32, 128)`, `y (32, 128)` 를 만든다. y 는 x 를 한 칸 민 것. 총 4,096 개의 "다음 토큰 맞히기" 문제.
2. `model(x, y)` → logits `(32, 128, 8192)` 와 loss 스칼라 (4,096 문제의 cross-entropy 평균).
3. `loss.backward()` → 6.8M 파라미터 각각의 `.grad`.
4. `clip_grad_norm_(1.0)` → 기울기 전체의 길이가 1 을 넘으면 비례해서 줄인다. 초기에는 자주 걸리고 나중에는 거의 안 걸린다.
5. `optimizer.step()` → AdamW 가 파라미터마다 이동 평균으로 보정한 걸음을 `lr` 배 만큼 옮기고, 행렬 파라미터는 `weight_decay` 만큼 0 쪽으로 당긴다.
6. 100 스텝마다 `estimate_loss` (train/val 각 8 배치 평균, 드롭아웃 끄고) → `log.jsonl` 한 줄, val 최저면 `best.pt`.

학습률 값 자체를 보자. 스텝 0 은 6e-4 × 1/100 = 6e-6 (거의 안 움직임), 스텝 100 에 6e-4 (최고), 이후 코사인으로 스텝 2,500 에 6e-5.
노트북 §1.2 의 왼쪽 그래프가 이 곡선이다.

## 7.3 CPU 에서 빠르게

**스레드 수.** 4장에서 측정한 함정: 하이퍼스레드까지 16 스레드로 돌리면 8 스레드보다 16배 느렸다. `setup_cpu()` 기본은
물리 코어 수(8)다. 스레드가 물리 코어보다 많으면 캐시·동기화 경쟁이 계산보다 커진다.

**스텝당 토큰 수 = batch_size × block_size.** 이것이 예산이다. `small-cpu` 는 32 × 128 = 4,096 토큰/스텝, 스텝당 약 1초
(8 스레드, 6.8M 파라미터). 2,500 스텝 = 45분, 1,000만 토큰 (평가·체크포인트 포함).

**D 드라이브는 한 번만 읽는다.** `scripts/train.py` 는 시작 시 코퍼스를 통째로 인코딩해 텐서로 메모리에 둔다.
학습 루프 안에서 `/mnt/d` 를 반복 읽지 않는다 (CLAUDE.md 함정).

## 7.4 과적합: 코퍼스가 작다는 것

코퍼스는 49만 BPE 토큰이다. 2,500 스텝이면 약 20 에폭, 6.8M 파라미터 모델은 문장을 외우기 시작한다. `small-cpu` 의
실제 결과 (2026-09-16, 8 스레드, 45분):

| step | train | val |
|---|---|---|
| 0 | 9.08 | 9.09 |
| 500 | 5.99 | 6.13 |
| 1000 | 5.40 | 5.76 |
| 2100 (best) | 4.77 | **5.485** |
| 2500 | 4.68 | 5.49 |

train 은 계속 내려가고 val 은 2,100 스텝에서 멈춘다. 1장 n-gram, 4장 MLP 와 같은 현상. 학습률이 코사인으로 거의
0 에 가까워진 덕에 val 이 크게 튀어 오르진 않았지만, 더 돌려도 좋아질 여지는 없다. perplexity 로는 exp(5.485) ≈ 241, 
8,192 개 중 240 개로 좁힌 셈이고, 글자당으로는 약 3.8 bit (9장).

버티는 수단은 `dropout: 0.2`, `weight_decay: 0.1`, 그리고 val 최저 시점의 `best.pt` 다. 근본 해법은 **데이터**다.
nanoGPT 의 Tiny Shakespeare 도 같은 크기(110만 자)에서 같은 곡선을 그린다. 이 프로젝트의 다음 단계는 코퍼스 2차 확장
(한국어 위키 일부)이며, 이 결정은 `docs/todo.md` 에 기록한다.

### 자바 개발자의 눈으로

| 이 장 | 자바로 치면 |
|---|---|
| `TrainConfig` dataclass + YAML | `@ConfigurationProperties` 로 바인딩되는 설정 클래스 |
| `Trainer.train()` | `while (step < max) { … }` 루프를 가진 서비스 객체. 상태(step, best_val, optimizer)를 필드로 |
| `torch.save({...}, path)` / `load_state_dict` | 직렬화. `state_dict` 는 `Map<String, Tensor>` 라 이름으로 매핑 |
| `log.jsonl` | 한 줄에 JSON 하나인 append-only 로그. `tail -f` 로 볼 수 있다 |
| `torch.no_grad()` 컨텍스트 | 이 블록 안에서는 그래프를 기록하지 않음 (평가·생성 때 메모리 절약) |
| `nohup … &` | 백그라운드 프로세스. 세션이 끊겨도 계속 돈다 |

### 흔한 오해

- **"학습률은 하나의 숫자"**, 스케줄이다. 같은 최고값이라도 warmup 없이 시작하면 초기에 망가지고, 감쇠 없이 끝내면 마지막에 요동친다.
- **"오래 돌릴수록 좋다"**, 데이터가 작으면 어느 시점부터 val 이 오른다. `best.pt` 가 그 전 시점을 잡아 두는 이유이고, 곡선을 보고 멈추는 것이 맞다.
- **"체크포인트에 모델만 있으면 이어서 학습할 수 있다"**, AdamW 의 이동 평균 상태(모델의 2배 크기)가 없으면 이어 학습이 처음 몇백 스텝 흔들린다. 그래서 `ckpt.pt` 는 옵티마이저까지 저장한다.
- **"CPU 스레드는 많을수록 빠르다"**, 물리 코어보다 많으면 오히려 느려진다(16배 측정). `setup_cpu()` 기본을 믿는다.

## 7.5 데이터를 늘리면 (ADR-0003, v1.1)

같은 모델(6.8M)을 근대문학 + 위키 혼합 코퍼스(624만 토큰, 13배)로 5,000 스텝(3,000만 토큰, 약 5 에폭) 학습한 `mixed-cpu`:

| | small-cpu | mixed-cpu |
|---|---|---|
| 학습 토큰 | 49만 (근대문학) | 624만 (근대문학 + 위키 673편) |
| 토크나이저 | bpe-8192 | bpe-8192-mixed (새로 학습) |
| 스텝 · 시간 | 2,500 · 45분 | 5,000 · 113분 |
| val 최저 시점 | 2,100 (이후 정체) | **5,000 (끝까지 내려감)** |
| 근대문학 val, 글자당 bit (9장) | 3.48 | **3.30** |

두 가지가 달라졌다. 첫째, val 이 마지막 스텝까지 내려간다. 데이터가 모델보다 커지니 외울 여지가 없다. 둘째, 위키를 섞었는데도
**근대문학 텍스트에서의 성능이 좋아졌다**(3.48 → 3.30 bit). 다른 도메인의 한국어가 문법·조사·어휘의 일반 지식을 보태 준 것이다.
대가는 생성 문체가 위키 쪽으로 기우는 것(8장 §5)과 토크나이저 교체로 이전 체크포인트와 호환되지 않는 것이다.

## 7.6 실습: 노트북 07

`notebooks/07-training.ipynb` (전체 실행 약 4분):

1. `configs/tiny-notebook.yaml` 로 600 스텝 학습, 로그·설정·체크포인트 파일 확인, 학습률 곡선
2. `read_log("small-cpu")` 로 긴 학습의 train/val 곡선과 best 지점
3. `best.pt` 에서 생성
4. 학습된 토큰 임베딩의 이웃 (3장 다시 보기)

긴 학습은 노트북 밖에서:

```bash
uv run python scripts/train.py --config configs/small-cpu.yaml            # 처음부터
uv run python scripts/train.py --config configs/small-cpu.yaml --resume   # 이어서
nohup uv run python -u scripts/train.py --config configs/small-cpu.yaml > data/runs/small-cpu.out 2>&1 &
```

## 스스로 점검

1. `small-cpu` 에서 1 에폭은 몇 스텝인가? 2,500 스텝은 몇 에폭인가?
2. warmup 이 없으면 무엇이 문제인가? cosine 감쇠가 없으면?
3. `best.pt` 와 `ckpt.pt` 의 차이와 각각의 용도는? 왜 `ckpt.pt` 는 모델의 3배 크기인가?
4. val loss 가 2,100 스텝에서 멈춘 것을 보고 할 수 있는 조치 세 가지를 우선순위대로 말하라.
5. `estimate_loss` 가 배치 여러 개를 평균하고 `model.eval()` 을 부르는 이유는?
6. 체크포인트에 토크나이저 이름을 기록하는 이유는?

## 관련 코드

- `src/shllm/train.py`, `TrainConfig` · `lr_at` · `Trainer` (`estimate_loss` · `save_checkpoint` · `resume` · `train`) · `load_checkpoint` · `load_yaml_config` · `read_log`
- `scripts/train.py`, 설정 파일로 긴 학습, `--resume` `--run-name` `--max-steps` `--threads`
- `configs/tiny-notebook.yaml` (노트북용 600 스텝) · `configs/small-cpu.yaml` (v1.0 본 학습)
- `tests/test_train.py`, 스케줄 모양 · 학습이 loss 를 줄이고 로그·체크포인트를 남김 · 되살리기·이어 학습 · YAML 파싱
