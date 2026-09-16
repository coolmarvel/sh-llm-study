---
title: 4장 — 신경망 기초: 세지 않고 배운다
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 4장 — 신경망 기초: 세지 않고 배운다

1장은 확률표를 **세어서** 만들었고, 3장은 임베딩 표의 값이 "학습으로 정해진다"고만 했다. 이 장의 질문은
**학습이란 정확히 무엇을 하는 것인가**다. 답은 세 단계다 — loss 를 정의하고, 파라미터에 대한 loss 의 **기울기**를
구하고, 기울기의 반대 방향으로 파라미터를 조금 옮긴다. 이 반복이 딥러닝의 전부이며, 7장의 학습 루프도
이것에 스케줄과 체크포인트를 붙인 것이다. 노트북 `notebooks/04-neural-network.ipynb`, 코드
`src/shllm/{autograd,numpy_lm,mlp}.py`.

## 4.1 기울기 — "이 값을 조금 바꾸면 loss 가 얼마나 변하나"

파라미터가 수백만 개인 함수의 최솟값을 찾는 유일한 현실적 방법은 **경사하강**이다. 각 파라미터 w 에 대해
∂loss/∂w (기울기) 를 구하면 "w 를 늘렸을 때 loss 가 얼마나 늘어나는가"를 알 수 있고, 그 반대 방향으로 조금
옮기면 loss 가 준다:

```
w ← w − lr · ∂loss/∂w        (lr = 학습률, 한 걸음의 크기)
```

문제는 기울기를 어떻게 구하느냐다. 손으로 미분하면 모델을 바꿀 때마다 다시 해야 한다. **자동미분(autograd)** 은
프로그램이 계산한 식을 기록해 두었다가 기계적으로 미분한다.

## 4.2 자동미분 — 연쇄법칙을 코드로 (`autograd.py`)

`Value` 는 숫자 하나에 두 가지를 붙인 것이다: 그 숫자가 **무엇으로부터 어떤 연산으로** 만들어졌는지(`_prev`, `_backward`),
그리고 최종 출력에 대한 편미분(`grad`). 연산자를 오버로드했으므로 보통의 산술식을 쓰면 그래프가 자동으로 생긴다.

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), "*")
    def _backward():
        self.grad  += other.data * out.grad     # d(a·b)/da = b
        other.grad += self.data  * out.grad     # d(a·b)/db = a
    out._backward = _backward
    return out
```

각 연산은 값 계산과 **국소 미분 규칙** 두 가지를 정의한다. `backward()` 는 그래프를 위상 정렬해 출력에서 입력 쪽으로
`_backward` 를 차례로 부른다 — 위에서 흘러온 `out.grad` 에 국소 미분을 곱해 아래로 넘기는 것이 **연쇄법칙**이다.
`+=` 인 이유는 한 값이 여러 곳에 쓰이면 기여가 쌓이기 때문이다 (`y = x·x + x` 이면 dy/dx = 2x + 1).

`torch.Tensor` 는 이것을 텐서 단위로, C++ 로, 수백 가지 연산에 대해 구현한 것이다. `requires_grad=True` 가 그래프
기록을 켜고, `.backward()` 가 `backward()`, `.grad` 가 `grad` 다. 노트북 §1 에서 같은 식의 미분값이 소수점 넷째 자리까지
일치하는 것을 확인한다. `src/shllm/autograd.py` 는 100줄이 안 된다 — 한 번 끝까지 읽어 두면 PyTorch 가 마법이 아니게 된다.

## 4.3 손으로 쓴 역전파 — 바이그램을 학습으로 (`numpy_lm.py`)

autograd 없이 NumPy 로 순전파·역전파를 직접 써 본다. 모델은 1장 바이그램의 학습 버전:

```
logits = W[x]                  (V, V) 표에서 현재 토큰의 행 — 다음 토큰 점수
p      = softmax(logits)       점수 → 확률 (exp 후 합으로 나눔)
loss   = −mean(log p[정답])     1장의 loss 와 같은 식 (cross-entropy)
```

역전파는 세 줄이다 (`backward`):

```python
dlogits = p.copy(); dlogits[range(N), y] -= 1    # d loss / d logits = p − onehot(y)
dlogits /= N                                       # mean 의 미분
np.add.at(dW, x, dlogits)                          # 룩업의 역전파 = 해당 행에 더하기
```

**softmax 와 cross-entropy 를 합치면 미분이 "예측 확률 − 정답"이 된다.** 정답 칸은 (p−1) 로 음수(점수를 올려라),
나머지는 p 로 양수(내려라). 딥러닝에서 가장 자주 쓰이는 사실이고, 6장 GPT 의 마지막 층에서도 똑같이 일어난다.

결과 (노트북 §2, 문자 토큰 V=2,827, 배치 4,096, lr 300):

| | loss |
|---|---|
| 시작 (W=0, 균등) | log V = 7.95 |
| 300 스텝 후 val | 3.58 |
| 1장 세어서 만든 표 val | 3.10 |

0 에서 출발한 W 가 세어서 만든 표 쪽으로 내려간다 (더 돌리면 계속 가까워진다). **세는 것과 배우는 것이 같은 곳에
도착한다.** 차이는 배우는 쪽이 "표"가 아닌 어떤 함수든 같은 절차로 다룰 수 있다는 것 — 그래서 다음 절부터는 표를 버린다.

**학습률.** lr 3 은 60 스텝에 7.5 (너무 느림), 300 은 4.8, 1,000 은 `inf` (발산). 학습률은 가장 민감한
하이퍼파라미터이며, 7장은 작게 시작해(warmup) 코사인으로 줄이는 스케줄을 쓴다.

**미니배치.** 매 스텝 코퍼스 전체가 아니라 무작위 4,096 곳만 본다. 기울기가 조금 잡음이 섞이지만 스텝이 훨씬 싸고,
잡음이 오히려 나쁜 지역 최솟값을 빠져나오게 돕는다. 이후 모든 학습은 미니배치다.

## 4.4 MLP 언어모델 — 문맥을 벡터로 (`mlp.py`)

Bengio et al. (2003) 의 모델. 어텐션 이전에 신경망으로 "다음 토큰 확률"을 만든 첫 구조다.

```
idx (B, T) ─임베딩 (V,C)─▶ (B, T, C) ─이어 붙이기─▶ (B, T·C) ─Linear+tanh─▶ (B, H) ─Linear─▶ (B, V)
```

```python
x = self.emb(idx)                          # (B, T, C)  3장 룩업
x = x.view(B, T * C)                       # 문맥 벡터들을 한 줄로
h = torch.tanh(self.hidden(x))             # (B, H)
logits = self.out(h)                       # (B, V)
loss = F.cross_entropy(logits, targets)    # softmax + −log p[정답] 을 한 번에, 수치적으로 안전하게
```

1장 n-gram 과의 결정적 차이 — **문맥이 dict 의 키가 아니라 벡터**다. ` 아버지가` 와 ` 어머니가` 는 n-gram 표에서
전혀 다른 행이지만, 임베딩이 비슷하면 은닉층 입력이 비슷해 한쪽에서 배운 것이 다른 쪽에 옮겨진다.

`train_steps` (가장 단순한 학습 루프):

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr)   # 파라미터마다 학습률을 자동 조절하는 SGD 의 개량판
for step in range(steps):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)   # 기울기는 누적되므로(Value 의 +=) 매번 비운다
    loss.backward()                   # autograd
    opt.step()                        # w ← w − lr·grad (Adam 식으로 보정해서)
```

결과 (노트북 §3, BPE 8,192 토큰, C=64, H=256, 3,000 스텝 × 배치 128, 스텝당 약 17ms):

| 모델 | 문맥 | val loss/토큰 |
|---|---|---|
| 2-gram (세기) | 1 | 7.67 |
| 5-gram (세기) | 4 | 8.46 |
| **MLP T=1** | 1 | **6.85** |
| MLP T=4 | 4 | 7.29 (train 6.01) |

- **같은 정보(직전 토큰 하나)인데 학습 쪽(MLP T=1)이 세기(2-gram)를 이긴다.** 임베딩 덕에 비슷한 토큰끼리 통계를 공유한다.
- MLP T=4 는 5-gram 을 크게 이기지만 T=1 보다는 아직 나쁘다. train 은 더 낮고 val 은 더 높다 — 1장에서 본 **과적합**.
  38만 토큰(코퍼스 한 바퀴 미만)만 보고 문맥 4개짜리 파라미터를 채우기엔 부족하다. 더 긴 학습, dropout, weight decay 가
  필요하고 7장에서 다룬다.
- n-gram 은 문맥을 늘리면 표가 비어 **나빠졌지만**, MLP 는 문맥을 벡터로 다루므로 본 적 없는 조합에도 **일반화**한다.

**MLP 의 한계** — 문맥 길이 T 가 고정이고, 자리마다 가중치가 따로다. "3칸 앞의 ` 아버지`" 와 "4칸 앞의 ` 아버지`" 는
`hidden` 의 서로 다른 열로 들어가 다른 취급을 받는다. 문맥을 늘리면 입력이 T·C 로 선형 증가한다. 5장의 어텐션은
문맥 길이가 가변이고, **위치가 아니라 내용으로** 어디를 볼지 정한다.

## 4.5 CPU 메모 — 스레드 수

이 장을 만들며 측정한 함정 하나: `torch.set_num_threads(16)` (하이퍼스레드 포함) 은 같은 MLP 스텝이 275ms, 8 스레드는
17ms 였다. 작은 행렬곱은 스레드끼리의 동기화·캐시 경쟁이 계산보다 크다. `setup_cpu()` 의 기본은 **물리 코어 수**
(`os.cpu_count() // 2`) 다 (`src/shllm/config.py`). 7장에서 다시 다룬다.

## 4.6 실습 — 노트북 04

`notebooks/04-neural-network.ipynb` (전체 실행 약 4분):

1. `Value` 로 미분, `torch` 와 대조
2. NumPy 바이그램: 순전파·역전파·SGD 300 스텝, 세어서 만든 표와 비교, 학습률 4종
3. `get_batch` 의 (x, y) 모양, MLP T=1·T=4 학습과 손실 곡선, n-gram 과 val 비교, 생성 미리보기

## 관련 코드

- `src/shllm/autograd.py` — `Value` (`__add__` `__mul__` `__pow__` `exp` `log` `tanh` `relu` `backward`)
- `src/shllm/numpy_lm.py` — `softmax` · `forward` · `backward` · `sgd_step`
- `src/shllm/mlp.py` — `NeuralBigram` · `MLPLanguageModel` · `train_steps`
- `src/shllm/data.py` — `get_batch` (x 와 한 칸 민 y)
- `tests/test_autograd.py` — torch 와 미분 일치 · 재사용 누적 · NumPy 기울기 ≡ torch · SGD 로 규칙 학습
- `tests/test_mlp.py` — 배치 모양 · 신경 바이그램 · MLP 가 2-문맥 규칙을 배우고 생성
