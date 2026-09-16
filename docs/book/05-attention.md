---
title: 5장 — 셀프 어텐션: 어디를 볼지 내용으로 정한다
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 5장 — 셀프 어텐션: 어디를 볼지 내용으로 정한다

4장 MLP 의 한계는 문맥 길이가 고정이고 자리마다 가중치가 따로라는 것이었다. 이 장의 질문은 **앞 토큰들 중 어디를 봐야
하는지 모델이 어떻게 아나**다. 답이 **셀프 어텐션** — 각 자리가 다른 자리들에 "시선"을 배분하고, 그 배분을 위치가 아니라
**내용**으로 계산한다. 노트북 `notebooks/05-attention.ipynb`, 코드 `src/shllm/attention.py`.

## 5.1 뼈대 — 문맥 섞기는 (T, T) 행렬곱이다

자리 t 의 출력을 "자리 0..t 의 평균"으로 만들고 싶다고 하자. 하삼각 행렬을 행마다 정규화하면 그것이 가중치다:

```
W = [[1,   0,   0  ],        W @ x  →  행 t = x[0..t] 의 평균
     [.5,  .5,  0  ],
     [.33, .33, .33]]
```

`(T, T)` 행렬 하나가 "누가 누구를 얼마나 보는가"를 정한다. 지금은 균등 평균이라 **자리만 보고** 정했다. 어텐션은 이
가중치를 **데이터에서** 계산한다 — 어떤 토큰이 어떤 토큰을 더 볼지, 벡터로.

## 5.2 Q · K · V — 부드러운 맵 조회

각 토큰 벡터 x (C 차원) 를 세 가지 역할로 투영한다 (각각 `nn.Linear(C, d)`):

| | 뜻 | 자바 비유 |
|---|---|---|
| **Q** (query) | 내가 찾는 것 | `map.get(key)` 의 인자 |
| **K** (key) | 나를 찾을 때 쓸 꼬리표 | 맵에 저장된 key |
| **V** (value) | 찾아지면 넘겨줄 내용 | 맵의 value |

`HashMap` 은 정확히 일치하는 key 하나를 찾지만, 어텐션은 **모든 key 와의 유사도로 부드럽게 섞는다**:

```
scores  = Q Kᵀ / √d          (T, T)  자리 i 의 질문 · 자리 j 의 꼬리표
weights = softmax(scores)    행마다 합 1 인 시선 배분
out     = weights V          (T, d)  본 만큼 내용을 섞는다
```

`attention()` 은 이 식 그대로다:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(d)     # (..., T, T)
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
return weights @ v, weights
```

**왜 √d 로 나누나.** d 차원 랜덤 벡터의 내적은 표준편차가 √d 에 비례해 커진다 (노트북 §2.1: d=512 에서 22.6). 큰 점수를
softmax 에 넣으면 한 칸에 0.999 로 몰려 기울기가 사라진다. √d 로 나누면 어느 d 에서든 분산이 1 근처다.

**인과(causal) 마스크.** 자리 i 는 i 이하만 봐야 한다. 미래 토큰을 보면 "다음 토큰 예측"이 답을 베끼는 것이 되기 때문이다.
`torch.tril` 하삼각을 True 로 두고, False 칸의 점수를 −∞ 로 채우면 softmax 후 정확히 0 이 된다.
`register_buffer` 로 두는 이유: 파라미터는 아니지만 모델과 함께 저장·이동돼야 하는 텐서.

## 5.3 어텐션은 순서를 모른다

마스크 없이 입력 순서를 섞으면 각 토큰의 출력은 그대로다 (`tests/test_attention.py::test_attention_is_order_invariant_without_positions`).
어텐션은 벡터의 **집합**을 섞는 연산이라 자리 정보가 없다. 그래서 3장 위치 임베딩을 입력에 **더해 둔** 것이다 —
Q·K 가 위치 벡터를 통해 "직전 자리"를 알아볼 수 있게.

## 5.4 멀티헤드 — 관계를 여러 개 동시에 (`CausalSelfAttention`)

헤드 하나는 시선 배분이 하나뿐이다. C 를 n_head 개로 나눠(각 head_size = C/n_head) 서로 다른 Q·K·V 를 두면, 한 헤드는
"직전 토큰"을, 다른 헤드는 "같은 주어"를 보는 식으로 관계를 병렬로 본다. 구현 요령:

```python
q, k, v = self.qkv(x).split(C, dim=-1)                               # Linear(C, 3C) 한 번으로 Q,K,V
q = q.view(B, T, n_head, head_size).transpose(1, 2)                  # (B, n_head, T, head_size)
out, weights = attention(q, k, v, mask)                              # 헤드 축은 배치처럼 브로드캐스팅
out = out.transpose(1, 2).reshape(B, T, C)                           # 헤드를 이어 붙여 C 로
return self.proj(out)                                                # 헤드들을 섞는 출력 투영
```

파라미터는 `qkv` 3C²+3C 와 `proj` C²+C, 합쳐 **약 4C²** — 6장 블록 파라미터 계산의 기초다.
`last_weights` (B, n_head, T, T) 를 남겨 두어 노트북과 대시보드가 어텐션 맵을 그린다.

## 5.5 정말 나은가 — 4장 MLP 와 같은 조건

문맥 8 토큰, 임베딩 64, 1,500 스텝 × 배치 64, 마지막 자리의 val loss (노트북 §4):

| 모델 | 파라미터 | val loss/토큰 |
|---|---|---|
| MLP T=8 (문맥 이어 붙이기) | 2.76M | 8.22 |
| 어텐션 한 층 + 은닉층 T=8 | 2.66M | **6.40** |

같은 스텝 수에서 차이가 크다. 이유 둘:

1. 어텐션 모델은 T 개 자리 **모두**에서 다음 토큰을 예측한다 (`(B, T, V)`). 같은 배치에서 8배 많은 학습 신호.
2. 문맥 어디에 있든 **같은 가중치**로 토큰을 본다. MLP 는 "3칸 앞" 과 "4칸 앞" 을 다른 가중치로 배워야 해서 데이터가
   8배 더 필요하고, 그래서 4장의 T=4 처럼 과적합했다.

## 5.6 실습 — 노트북 05

`notebooks/05-attention.ipynb` (전체 실행 약 3분):

1. 하삼각 평균 행렬로 "문맥 섞기 = 행렬곱" 확인
2. Q·K·V 를 한 줄씩 계산하고 `attention()` 과 일치 확인, √d 실험
3. 임베딩 → `CausalSelfAttention` 의 shape 흐름, 가중치 `(B, 헤드, T, T)`, 파라미터 수
4. MLP vs 어텐션 한 층 비교 학습

## 관련 코드

- `src/shllm/attention.py` — `attention` · `causal_mask` · `AttentionHead` · `CausalSelfAttention`
- `tests/test_attention.py` — 공식 ≡ 루프 계산 · 순서 불변 · 인과성(미래 변경이 과거 출력에 영향 없음) · 멀티헤드 모양·파라미터
