---
title: 1장 — 글자를 숫자로, 확률표로 글쓰기
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 1장 — 글자를 숫자로, 확률표로 글쓰기

이 장의 질문은 두 개다. **글자를 어떻게 숫자로 바꾸나?** 그리고 **"다음 글자 확률표" 하나만으로 글이 써지나?**
노트북 `notebooks/01-char-tokenizer-bigram.ipynb` 를 열고 함께 읽는다. 코드는 `src/shllm/tokenizer.py`(`CharTokenizer`)
와 `src/shllm/bigram.py`(`BigramModel`, `NGramModel`) 두 파일이 전부다.

이 장이 끝나면 0장의 정의 — "LLM 은 P(다음 토큰 | 지금까지의 토큰들) 을 계산하는 함수" — 를 가장 원시적인 방법으로
**직접 구현한 상태**가 된다. 이후 장들은 이 함수를 더 잘 만드는 방법일 뿐이다.

## 1.1 토큰화 — 모델은 문자열을 모른다

신경망은 숫자만 받는다. 텍스트를 정수 열로 바꾸는 규칙이 **토크나이저**이고, 정수 하나가 **토큰**, 토큰의 종류 수가
**어휘 크기 V** 다. 가장 단순한 규칙은 "글자 하나 = 토큰 하나":

```
"옛날 옛적에"  →  [2237, 1421, 1, 2237, 2340, 2213]
```

`CharTokenizer` (`src/shllm/tokenizer.py`) 는 코퍼스에 등장한 글자를 정렬해 0 부터 번호를 붙인다.

```python
class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string → int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # int → string

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
```

자바 개발자에게 낯선 부분 둘:

- `{ch: i for i, ch in enumerate(chars)}` 는 **딕셔너리 컴프리헨션** — `for` 루프로 `Map` 을 채우는 것을 한 줄로 쓴 것.
  `enumerate` 는 `(인덱스, 원소)` 쌍을 돌려준다.
- `[self.stoi[ch] for ch in text]` 는 **리스트 컴프리헨션** — `text.chars().map(stoi::get).toList()` 에 해당.

계약 하나: **학습에 쓴 토크나이저와 생성에 쓰는 토크나이저는 같은 것이어야 한다.** 번호가 하나라도 밀리면 모델 출력은
전부 엉뚱한 글자가 된다. 그래서 `save()`/`load()` 로 `data/tokenizers/char.json` 에 저장하고, 이후 장은 이 파일을 읽는다.

## 1.2 한글 어휘의 모양 — 긴 꼬리

코퍼스 114만 자의 어휘를 세면 이렇다 (노트북 §1.1).

| 종류 | 수 |
|---|---|
| 한글 음절 | 1,619 (가능한 11,172 중) |
| 한자 | 1,088 |
| 공백·기호·영문 | 120 |
| **합계 V** | **2,827** |

그런데 가장 흔한 100자가 전체 텍스트의 76%, 1,000자가 99.3% 를 차지하고, 662자는 딱 한 번 나온다.
**어휘의 대부분은 거의 쓰이지 않는 글자**다. 이 "긴 꼬리"는 이 장 내내 문제를 일으킨다 — 한 번도 못 본 글자 조합을
모델이 어떻게 다룰 것인가 (1.6 스무딩). 영어(V ≈ 65)에는 없는, 한글 고유의 사정이다. 2장의 BPE 는 이 어휘 단위 자체를 바꾼다.

## 1.3 언어모델 = 조건부 확률, 가장 단순한 구현 = 세어서 나누기

P(다음 글자 | 직전 글자) 를 구하는 가장 정직한 방법은 코퍼스에서 **세는 것**이다.

```
P("는" | "호") = ("호" 다음에 "는" 이 온 횟수) / ("호" 가 앞자리에 온 총 횟수)
```

직전 글자 하나만 보는 모델을 **바이그램(bigram)** 이라 한다. 확률표는 `(V, V)` 행렬 하나 — 행이 현재 글자, 열이 다음 글자,
각 행의 합이 1. V=2,827 이면 800만 칸이고 텐서 하나에 들어간다.

`BigramModel.fit` (`src/shllm/bigram.py`) 은 이 표를 파이썬 루프 없이 채운다:

```python
ids = torch.as_tensor(ids)                 # (N,)
prev, nxt = ids[:-1], ids[1:]              # 같은 열을 한 칸 어긋나게 겹치면 모든 인접 쌍이 나온다
self.counts.index_put_((prev, nxt), torch.ones_like(prev), accumulate=True)
```

`index_put_(..., accumulate=True)` 는 "`counts[prev[i], nxt[i]] += 1` 을 모든 i 에 대해" 다. `accumulate=True` 가 없으면
같은 쌍이 여러 번 나올 때 1 로 **덮어써** 버린다 — 흔한 함정. 102만 쌍을 세는 데 40ms 다.

확률표는 행마다 나눈다. `(V, V) / (V, 1)` — 크기가 다른 텐서끼리 나누면 작은 쪽이 늘어나 맞춰지는 **브로드캐스팅**이다.

```python
p = self.counts.float() + self.smoothing
self._probs = p / p.sum(dim=1, keepdim=True)   # (V, V) / (V, 1)
```

`keepdim=True` 를 빼면 `(V,)` 가 되어 엉뚱한 축으로 브로드캐스팅된다. shape 주석을 습관적으로 다는 이유다.

## 1.4 생성 — 확률에서 주사위 던지기

확률표가 있으면 생성은 반복문 하나다 (`BigramModel.generate`):

```python
for _ in range(max_new_tokens):
    p = self.next_token_probs(out)                       # (V,)  직전 토큰의 행
    out.append(int(torch.multinomial(p, num_samples=1, generator=generator)))
```

`torch.multinomial(p, 1)` 은 확률 벡터 `p` 대로 인덱스 하나를 뽑는다 — 면이 V 개이고 각 면의 확률이 다른 주사위.
가장 큰 확률만 고르면(argmax) 같은 문장이 반복되므로 **뽑는다**. 8장의 temperature·top-k 는 이 주사위를 어떻게 깎을지의 문제다.
`torch.Generator().manual_seed(1337)` 을 넘기면 같은 시드에서 같은 문장이 나온다 — 실험을 재현하려면 필수.

결과 (노트북 §2.1): `옛輸않는 주 집 요, 있는 하는가 죽어남편이따로 아, 가 이 꼬옥도 기고 …`.
두세 글자 단위("하는가", "있는")는 그럴듯하지만 문장이 되지 않는다. 직전 한 글자밖에 못 보니 당연하다.

## 1.5 평가 — loss 하나로 "얼마나 좋은가"

생성문을 눈으로 보는 것은 비교가 안 된다. 숫자가 필요하다. 모델이 실제 텍스트의 **매 글자에 준 확률**을 log 로 바꿔
평균하고 부호를 뒤집은 것이 **loss** (평균 음의 로그가능도, negative log-likelihood):

```
loss = -(1/N) Σ log P(ids[i+1] | ids[i])
```

정답에 확률 1 을 주면 log 1 = 0, 확률을 낮게 줄수록 커진다. 이 값이 4장부터 나올 **cross-entropy** 와 같은 양이며,
7장의 학습 루프는 이 값을 줄이는 것이 전부다. 코드는 두 줄이다 (`BigramModel.loss`):

```python
logp = torch.log(self.probs[ids[:-1], ids[1:]])   # (N-1,)  실제 나온 쌍마다 log 확률
return float(-logp.mean())
```

`probs[ids[:-1], ids[1:]]` 는 **팬시 인덱싱** — 행 인덱스 배열과 열 인덱스 배열을 동시에 주면 각 쌍의 값을 벡터로 뽑는다.

`exp(loss)` 는 **perplexity** — "매 순간 몇 개 중 하나를 찍는 셈인가". 기준점 둘:

| 모델 | val loss | perplexity |
|---|---|---|
| 아무 정보 없이 균등하게 찍기 | log V = 7.95 | 2,827 |
| 바이그램 | 3.17 | 23.8 |

균등은 2,827 개 중 하나, 바이그램은 24 개 중 하나를 찍는 셈이다. 100배 이상 좋아졌다.

**train / val 분리.** 전체를 정수 열로 바꾼 뒤 앞 90% 로 표를 만들고(train), 뒤 10%(val) 에서 loss 를 잰다. val 은
모델이 못 본 텍스트라 "처음 보는 글에서의 성능"을 재는 것이고, 이후 모든 장에서 이 분리를 유지한다. 시험 문제를
미리 보여 주면 점수가 의미 없어지는 것과 같다.

## 1.6 스무딩 — 한 번도 못 본 쌍

`smoothing=0` 이면 학습에 없던 쌍이 val 에 나오는 순간 log 0 = −∞ 로 loss 가 터진다 (노트북 §2.3 에서 `inf`).
해법은 모든 칸에 **가짜 횟수 α** 를 미리 더해 두는 것(가산 스무딩)이다. 교과서는 α=1 (라플라스) 을 권하지만, V=2,827
이면 **행마다 가짜 횟수 2,827 개**가 생겨 실제 횟수가 수십 개인 행을 묻어 버린다. 노트북 §2.3 의 결과:

| α | train | val | 생성문 |
|---|---|---|---|
| 0 | 3.17 | ∞ | 좋음 (그러나 평가 불가) |
| 1 | 3.84 | 3.69 | 한자 범벅 — 균등분포에 가까워짐 |
| 0.1 | 3.35 | 3.26 | |
| **0.01** | 3.21 | **3.17** | 기본값 |
| 0.001 | 3.18 | 3.19 | 너무 작아 못 본 쌍의 벌점이 과함 |

α 를 val loss 가 가장 낮은 값으로 고른다. **하이퍼파라미터는 val 로 고른다** — 이 패턴은 앞으로 학습률, 모델 크기, 문맥 길이 등
모든 결정에 반복된다.

## 1.7 문맥을 늘리면 — n-gram 과 표의 폭발

직전 1글자가 아니라 n−1 글자를 보면 당연히 더 잘 맞힌다. 문제는 표의 크기: 가능한 문맥이 V^(n−1) 가지다.

| n | 가능한 문맥 | 실제 등장한 문맥 |
|---|---|---|
| 2 | 2,827 | 2,791 |
| 3 | 8.0 × 10⁶ | 46,198 |
| 4 | 2.3 × 10¹⁰ | 175,842 |
| 5 | 6.4 × 10¹³ | 411,442 |

n=3 만 돼도 텐서에 안 들어간다. `NGramModel` 은 **실제 등장한 문맥만** dict 에 담는다 — 자바로 치면
`Map<List<Integer>, Map<Integer, Integer>>`. 파이썬의 `defaultdict` 는 없는 키를 읽으면 기본값을 만들어 넣는
`Map.computeIfAbsent` 다. 문맥은 튜플(불변 리스트)이라 dict 키가 된다.

**스무딩을 다르게 해야 한다.** 1.6 처럼 균등분포에 섞으면 n=4 에서는 문맥 대부분이 한두 번밖에 안 나와 가짜 횟수가
실제를 압도하고, 생성이 즉시 "아무 글자나"로 무너진다 (이 장을 만들며 실제로 겪은 일이다). 대신 **한 단계 짧은 문맥의 분포**에 섞는다:

```
P_k(t | 문맥 k개) = ( count_k(문맥, t) + α · P_{k-1}(t | 문맥 k-1개) ) / ( total_k(문맥) + α )
P_0(t)            = ( count_0(t) + α / V ) / ( N + α )                 ← 유니그램은 균등분포와 섞는다
```

긴 문맥을 한 번도 못 봤으면(total=0) 자동으로 짧은 문맥의 답이 되고, 한두 번 본 문맥이면 그 횟수와 짧은 문맥의 답이
α 만큼의 비율로 섞인다 (**보간, interpolation**). 어휘 크기와 무관하고, 재귀 한 번으로 구현된다 (`NGramModel._probs_vec`):

```python
def _probs_vec(self, tokens, k):                          # 문맥 k개짜리 확률 벡터 (V,)
    base = uniform if k == 0 else self._probs_vec(tokens, k - 1)
    p = self.smoothing * base                             # 가짜 횟수 α 개를 짧은 문맥의 분포 모양으로
    for tok, c in self.counts[k].get(ctx, {}).items():
        p[tok] += c
    return p / (self.totals[k].get(ctx, 0) + self.smoothing)
```

그래서 `fit` 은 문맥 길이 0 ~ n−1 의 표를 **전부** 센다(표가 n 개).

결과 (노트북 §3, α=1):

| n | train loss | val loss | 생성문 |
|---|---|---|---|
| 1 | 4.79 | 4.70 | 글자 빈도대로 아무렇게나 |
| 2 | 3.18 | 3.20 | 두세 글자 단위만 그럴듯 |
| 3 | 2.35 | **2.88** | 어절이 살아남 — `"연구할 테요?"` |
| 4 | 1.62 | 3.14 | 문장처럼 보임 |
| 5 | 1.04 | 3.50 | 가장 자연스러움 — 그러나 |

두 곡선이 n=3 에서 갈라진다. **train loss 는 계속 내려가고 val loss 는 도로 오른다.** 문맥이 길수록 학습 텍스트를
"외우는" 것이지 언어를 배우는 것이 아니다 — **과적합(overfitting)**. 원인은 표가 비어 있어서다(**희소성, sparsity**):
n=5 에서 채워진 문맥은 가능한 문맥의 10⁻⁸ 이다. 생성문 안에서 코퍼스에 그대로 있는 최장 조각을 찾아 보면 n 이 커질수록
길어진다 — 생성문은 코퍼스 조각을 이어 붙인 것에 가까워진다.

## 1.8 왜 신경망인가 — 이 장의 결론

n-gram 표의 근본 한계는 **문맥이 dict 의 키**라는 것이다. `"호랑이가"` 와 `"호랑이는"` 은 뜻이 거의 같은데 표에서는
전혀 다른 행이라 한쪽에서 센 것이 다른 쪽에 조금도 도움이 안 된다. 문맥이 조금만 달라도 처음부터 다시 세야 하니
V^(n−1) 칸을 채울 데이터는 영원히 부족하다.

해결의 방향은 **문맥을 키가 아니라 숫자 벡터로** 다루는 것이다. 비슷한 문맥이 비슷한 벡터가 되면, 한 문맥에서 배운 것이
이웃 문맥으로 **일반화**된다. 그것이 3장(임베딩: 토큰 → 벡터)과 4장(신경망: 벡터 → 확률, 세지 않고 학습)이 하는 일이고,
5장의 어텐션은 "문맥의 어느 부분을 볼지"까지 학습한다. 그 모든 모델이 이 장과 똑같이 세 가지를 한다 —
**확률 계산(`next_token_probs`) · 생성(`generate`) · 평가(`loss`)**. 인터페이스는 안 바뀌고 안쪽만 바뀐다.

## 1.9 더 읽을 것 — 스무딩의 세계

이 장의 보간 α 는 1 로 고정했지만, 1장을 만들며 잰 값으로는 α=3 이 n≥3 에서 val 이 더 낮았다 (3-gram 2.88 → 2.81,
5-gram 3.50 → 3.13). α 도 하이퍼파라미터이므로 1.6 과 같은 방법 — val 로 고른다 — 을 적용하면 된다. 노트북 §3 의
`NGramModel(k, V, smoothing=α)` 에서 α 를 바꿔 직접 확인해 보라.

더 나은 스무딩이 있다. **Kneser-Ney**: 짧은 문맥으로 물러날 때 "그 토큰이 얼마나 자주 나왔나"가 아니라 "**얼마나 다양한
문맥 뒤에** 나왔나"를 센다. 예를 들어 "샌"은 "프란시스코" 앞에만 나오므로 빈도는 높아도 새로운 문맥에서는 확률을 낮게
줘야 한다. n-gram 시대의 최종 형태이며, 신경망 이전 언어모델의 기준선이었다 (Chen & Goodman 1999). 이 교재는 여기서
n-gram 을 떠나지만, 3장 이후의 모델이 "왜 더 나은가"를 잴 때 이 기준선을 기억해 두면 좋다.

## 1.10 실습 — 노트북 01

`notebooks/01-char-tokenizer-bigram.ipynb` (전체 실행 약 1분):

1. 코퍼스 → `CharTokenizer` → `data/tokenizers/char.json` 저장, 어휘 구성(한글/한자/기타)과 긴 꼬리
2. train/val 분리 → `BigramModel` 횟수표·확률표, `'호'` 다음 글자 top-8, 흔한 40자 확률표 히트맵
3. 생성, loss·perplexity, 스무딩 α 비교 표
4. `NGramModel` n=1~5 의 train/val loss 곡선, 생성문, 코퍼스에서 베낀 최장 조각, 표의 채워진 비율

## 관련 코드

- `src/shllm/tokenizer.py` — `CharTokenizer` (`from_text` · `encode` · `decode` · `save` · `load`)
- `src/shllm/bigram.py` — `BigramModel` ((V, V) 텐서, 가산 스무딩) · `NGramModel` (dict, 짧은 문맥으로 보간)
- `tests/test_bigram.py` — 횟수·확률 합·생성 재현성·스무딩·보간 후퇴·긴 문맥이 더 잘 맞히는지
- `src/shllm/config.py` — `TOKENIZER_DIR` (`data/tokenizers/`)
