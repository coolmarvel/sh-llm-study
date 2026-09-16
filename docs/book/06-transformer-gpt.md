---
title: 6장 — Transformer 블록 → GPT 조립
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 6장 — Transformer 블록 → GPT 조립

5장의 어텐션 한 층만으로도 MLP 를 이겼다. 이 장의 질문은 **어텐션을 어떻게 쌓아 GPT 가 되나**다. 답은 블록 하나를
정의하고 그것을 반복하는 것인데, 그냥 쌓으면 학습이 안 된다 — **LayerNorm** 과 **잔차 연결**이 그래서 있다.
노트북 `notebooks/06-transformer-gpt.ipynb`, 코드 `src/shllm/model.py`. 구조는 GPT-2 / nanoGPT 와 같다.

## 6.1 블록 — 소통 한 번, 계산 한 번

```python
class Block(nn.Module):
    def forward(self, x):                  # (B, T, C) → (B, T, C)
        x = x + self.attn(self.ln1(x))     # 토큰들 사이의 소통 (5장)
        x = x + self.mlp(self.ln2(x))      # 토큰 각자의 계산
        return x
```

입력과 출력 shape 이 같으므로 몇 개든 쌓을 수 있다. 어텐션이 다른 자리의 정보를 **모아 오고**, MLP 가 그것을 자리마다
**가공**한다. MLP 는 `C → 4C → C` 두 층에 GELU (ReLU 의 부드러운 판) — 4장 MLP 와 모양은 같지만 문맥을 이어 붙이지 않는다.
문맥은 어텐션이 이미 섞었다.

## 6.2 LayerNorm — 벡터마다 눈금 맞추기

토큰 벡터 하나 (C 차원) 안에서 평균 0·표준편차 1 로 맞춘 뒤, 학습되는 배율 γ 와 이동 β 를 적용한다:

```
y = γ · (x − mean(x)) / std(x) + β        (mean, std 는 C 차원 안에서)
```

배치나 이웃 토큰과 무관하게 **각 벡터 스스로** 정규화하므로 문장 길이·배치 크기에 영향받지 않는다 (BatchNorm 과의 차이).
층을 거칠수록 값의 크기가 제멋대로 커지는 것을 막아 깊은 모델의 학습을 가능하게 한다. 어텐션·MLP **앞에** 두는
Pre-LN 방식이 GPT-2 의 선택이고 학습이 더 안정적이다.

## 6.3 잔차 연결 — 기울기의 고속도로

`y = f(x)` 대신 `y = x + f(x)` 로 쓰면 미분에 항상 항등항 1 이 있다. 노트북 §2 의 실험: tanh 층 32개를 잔차 없이
쌓으면 입력에 도착하는 기울기가 사실상 0 (기울기 소실), 잔차가 있으면 층 수와 무관하게 살아 있다.

깊은 GPT 에서 잔차 스트림 x 는 "토큰의 현재 상태"를 실어 나르는 버스이고, 각 블록은 거기에 **얼마를 더할지**만 배운다.
GPT-2 는 잔차로 들어가는 투영(`proj`)을 `0.02/√(2L)` 로 더 작게 초기화해 층이 더해질수록 분산이 커지는 것을 막는다
(`GPT.__init__`).

## 6.4 GPT 조립

```
idx (B, T) ─GPTEmbedding─▶ (B, T, C) ─Block × L─▶ (B, T, C) ─LayerNorm─▶ ─lm_head (C→V)─▶ logits (B, T, V)
```

```python
def forward(self, idx, targets=None):
    x = self.drop(self.embed(idx))          # 3장
    for block in self.blocks:
        x = block(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)                # (B, T, V) — 모든 자리에서 다음 토큰 점수를 한 번에
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets is not None else None
    return logits, loss
```

두 가지 관례:

- **가중치 묶기(weight tying)**: 입력 임베딩 (V, C) 와 출력 `lm_head` (V, C) 를 같은 텐서로 쓴다
  (`self.lm_head.weight = self.embed.tok.weight`). "토큰 → 벡터" 와 "벡터 → 토큰 점수" 가 같은 표를 공유하는 것이 자연스럽고,
  파라미터 V·C 를 아낀다.
- **드롭아웃**: 학습 중 일부 활성값을 무작위로 0 으로 만든다. 코퍼스가 작을수록 필요하다 (7장 `dropout: 0.2`).
  `model.eval()` 이 끄고 `model.train()` 이 켠다 — 평가·생성 전에 `eval()` 을 잊으면 결과가 흔들린다.

## 6.5 파라미터는 어디에 있나

| 부품 | 식 | small-cpu (C=256, L=6, V=8192) |
|---|---|---|
| 토큰 임베딩 (= lm_head) | V·C | 2,097,152 |
| 블록당 어텐션 | ≈ 4C² | 263,168 |
| 블록당 MLP | ≈ 8C² | 525,568 |
| 블록 합계 | ≈ 12·C²·L | 4,730,880 |
| **총계** (위치 임베딩 제외) | | **6,836,224** |

GPT-2 small (C=768, L=12, V=50,257) 은 같은 식으로 1억 2,400만. **구조는 같고 숫자만 다르다.** 우리 모델은 어휘가
작고 폭이 좁아 CPU 로 수 시간에 학습할 수 있는 크기다.

## 6.6 짧은 학습과 어텐션 맵

노트북 §4 에서 2층·C=128 모델(1.4M)을 600 스텝만 돌려 본다 — 약 4분에 val 6.1 로, 5장의 어텐션 한 층(6.4)보다 낫다. 학습 후 `model.attention_maps()` 가 층별 `(B, n_head, T, T)` 를
돌려주고, 그 히트맵에서 헤드마다 다른 패턴이 보인다 — 직전 토큰만 보는 헤드, 문장 첫 토큰에 몰리는 헤드(정보가 없을 때의
"쉼터"), 넓게 퍼지는 헤드. 상삼각이 0 인 것은 인과 마스크다. 본 학습(7장) 뒤 대시보드에서 같은 그림을 본다.

## 6.7 실습 — 노트북 06

`notebooks/06-transformer-gpt.ipynb` (전체 실행 약 4분):

1. LayerNorm 전후의 평균·표준편차, γ·β
2. 잔차 유무에 따른 32층 기울기 크기
3. `GPT` 조립, `(B, T) → (B, T, V)`, 초기 loss ≈ log V, 세 크기의 파라미터 표
4. 600 스텝 학습, 층·헤드별 어텐션 맵, 생성 미리보기

## 관련 코드

- `src/shllm/model.py` — `GPTConfig` · `MLP` · `Block` · `GPT` (`forward` · `n_params` · `attention_maps`)
- `src/shllm/attention.py` — `CausalSelfAttention` (5장), `src/shllm/embedding.py` — `GPTEmbedding` (3장)
- `tests/test_model.py` — 블록 shape 유지 · 파라미터 수 식 · 가중치 묶기 · 인과성 · 규칙 학습 · block_size 초과
