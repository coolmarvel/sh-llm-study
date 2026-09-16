---
title: 6장 Transformer 블록 → GPT 조립
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 6장 Transformer 블록 → GPT 조립

5장의 어텐션 한 층만으로도 MLP 를 이겼다. 이 장의 질문은 **어텐션을 어떻게 쌓아 GPT 가 되나**다. 답은 블록 하나를
정의하고 그것을 반복하는 것인데, 그냥 쌓으면 학습이 안 된다. **LayerNorm** 과 **잔차 연결**이 그래서 있다.
노트북 `notebooks/06-transformer-gpt.ipynb`, 코드 `src/shllm/model.py`. 구조는 GPT-2 / nanoGPT 와 같다.

## 읽기 전에

- 5장의 `CausalSelfAttention` (입력 `(B,T,C)` → 출력 `(B,T,C)`), 4장의 MLP 와 cross-entropy, 3장의 `GPTEmbedding`.
- **평균과 표준편차**: 벡터 값들의 중심과 퍼짐. 표준화 = (값 − 평균) / 표준편차 → 평균 0, 표준편차 1.
- 파라미터 수 세기: `nn.Linear(a, b)` 는 a·b + b 개.

## 핵심 용어

| 용어 | 한 줄 정의 | 비유 |
|---|---|---|
| **블록(Block)** | 어텐션 + MLP 한 세트. 입력과 출력 shape 이 같아 쌓을 수 있다 | 같은 인터페이스를 가진 필터를 파이프라인에 연결 |
| **LayerNorm** | 토큰 벡터 하나 안에서 표준화 후 γ·x+β | 센서마다 눈금을 0~1 로 맞추기 |
| **잔차 연결(residual)** | `x + f(x)`. 부품은 "고칠 양"만 배운다 | 원본은 그대로 두고 diff 만 적용 |
| **잔차 스트림** | 블록들을 관통하며 계속 더해지는 x | 버스에 실린 토큰의 현재 상태 |
| **Pre-LN** | LayerNorm 을 어텐션·MLP **앞에** 두는 배치 (GPT-2) | 입력 검증을 먼저 |
| **GELU** | ReLU 의 부드러운 판. 음수를 살짝 남긴다 | 문턱이 둥근 스위치 |
| **드롭아웃** | 학습 중 값 일부를 무작위로 0 으로 | 팀원 몇 명이 무작위로 빠져도 일하게 훈련 |
| **weight tying** | 입력 임베딩과 출력 행렬을 같은 텐서로 | 인코더·디코더가 같은 사전을 공유 |
| **lm_head** | 마지막 `(C → V)` 선형층. 벡터를 어휘 점수로 | 최종 분류기 |
| **n_layer / n_head / n_embd** | 블록 수 / 헤드 수 / 폭 C. 모델 크기를 정하는 세 숫자 | 깊이 · 관점 수 · 폭 |

## 6.1 블록: 소통 한 번, 계산 한 번

```python
class Block(nn.Module):
    def forward(self, x):                  # (B, T, C) → (B, T, C)
        x = x + self.attn(self.ln1(x))     # 토큰들 사이의 소통 (5장)
        x = x + self.mlp(self.ln2(x))      # 토큰 각자의 계산
        return x
```

입력과 출력 shape 이 같으므로 몇 개든 쌓을 수 있다. 어텐션이 다른 자리의 정보를 **모아 오고**, MLP 가 그것을 자리마다
**가공**한다. MLP 는 `C → 4C → C` 두 층에 GELU (ReLU 의 부드러운 판), 4장 MLP 와 모양은 같지만 문맥을 이어 붙이지 않는다.
문맥은 어텐션이 이미 섞었다.

## 6.2 LayerNorm: 벡터마다 눈금 맞추기

토큰 벡터 하나 (C 차원) 안에서 평균 0·표준편차 1 로 맞춘 뒤, 학습되는 배율 γ 와 이동 β 를 적용한다:

```
y = γ · (x − mean(x)) / std(x) + β        (mean, std 는 C 차원 안에서)
```

배치나 이웃 토큰과 무관하게 **각 벡터 스스로** 정규화하므로 문장 길이·배치 크기에 영향받지 않는다 (BatchNorm 과의 차이).
층을 거칠수록 값의 크기가 제멋대로 커지는 것을 막아 깊은 모델의 학습을 가능하게 한다. 어텐션·MLP **앞에** 두는
Pre-LN 방식이 GPT-2 의 선택이고 학습이 더 안정적이다.

### 단계별 예시: LayerNorm 손계산

토큰 벡터 x = [1, 2, 3, 10] (C=4). 평균 = 4, 분산 = ((−3)²+(−2)²+(−1)²+6²)/4 = 50/4 = 12.5, 표준편차 = 3.54.
표준화: (x − 4) / 3.54 = **[−0.85, −0.57, −0.28, 1.70]**. 평균 0, 표준편차 1 이 됐고, 10 이 튀던 값이 1.7 로 눌렸다.
γ=[1,1,1,1], β=[0,0,0,0] 이면 이것이 출력이다. 학습이 γ·β 를 바꾸면 "이 채널은 두 배로, 저 채널은 +0.3" 같은 조정이 더해진다.
두 번째 토큰 [0.1, 0.2, 0.3, 0.4] 는 자기 평균 0.25, 표준편차 0.112 로 따로 표준화된다. 첫 토큰의 값과 무관하다. 이것이 "벡터마다" 의 뜻이다.

노트북 §1 이 같은 두 벡터를 `nn.LayerNorm(4)` 에 넣어 위 값을 그대로 출력한다.

## 6.3 잔차 연결: 기울기의 고속도로

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
    logits = self.lm_head(x)                # (B, T, V), 모든 자리에서 다음 토큰 점수를 한 번에
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1)) if targets is not None else None
    return logits, loss
```

두 가지 관례:

- **가중치 묶기(weight tying)**: 입력 임베딩 (V, C) 와 출력 `lm_head` (V, C) 를 같은 텐서로 쓴다
  (`self.lm_head.weight = self.embed.tok.weight`). "토큰 → 벡터" 와 "벡터 → 토큰 점수" 가 같은 표를 공유하는 것이 자연스럽고,
  파라미터 V·C 를 아낀다.
- **드롭아웃**: 학습 중 일부 활성값을 무작위로 0 으로 만든다. 코퍼스가 작을수록 필요하다 (7장 `dropout: 0.2`).
  `model.eval()` 이 끄고 `model.train()` 이 켠다. 평가·생성 전에 `eval()` 을 잊으면 결과가 흔들린다.

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

노트북 §4 에서 2층·C=128 모델(1.4M)을 600 스텝만 돌려 본다. 약 4분에 val 6.1 로, 5장의 어텐션 한 층(6.4)보다 낫다. 학습 후 `model.attention_maps()` 가 층별 `(B, n_head, T, T)` 를
돌려주고, 그 히트맵에서 헤드마다 다른 패턴이 보인다. 직전 토큰만 보는 헤드, 문장 첫 토큰에 몰리는 헤드(정보가 없을 때의
"쉼터"), 넓게 퍼지는 헤드. 상삼각이 0 인 것은 인과 마스크다. 본 학습(7장) 뒤 대시보드에서 같은 그림을 본다.

### 데이터 흐름을 shape 로 따라가기 (small-cpu, B=32, T=128, C=256, V=8192)

| 단계 | 코드 | shape |
|---|---|---|
| 입력 토큰 | `idx` | (32, 128) 정수 |
| 임베딩 | `embed(idx)` | (32, 128, 256) |
| 블록 × 6 | `block(x)` | (32, 128, 256) 유지 |
| 최종 정규화 | `ln_f(x)` | (32, 128, 256) |
| 어휘 점수 | `lm_head(x)` | (32, 128, 8192) |
| loss | `cross_entropy(logits.view(-1, 8192), targets.view(-1))` | 4,096 개 자리의 평균 → 스칼라 |

`(B, T, V)` 가 나온다는 것은 **한 번의 forward 로 T 개 자리에서 각각 다음 토큰을 예측**한다는 뜻이다. 자리 t 의 정답은 `targets[:, t] = idx[:, t+1]` (`get_batch` 의 한 칸 민 y). 인과 마스크 덕에 자리 t 의 예측은 t 이하만 보고 만들어졌으므로 컨닝이 아니다. 1장 바이그램이 쌍 하나씩 세던 일을 GPT 는 배치 전체에 대해 한 번에 한다.

### 자바 개발자의 눈으로

| 이 장 | 자바로 치면 |
|---|---|
| `nn.ModuleList([Block(cfg) for _ in range(L)])` | `List<Block>` 필드. `for (Block b : blocks) x = b.forward(x)` |
| `x = x + self.attn(self.ln1(x))` | `x = add(x, attn.forward(ln1.forward(x)))`. 새 배열을 만들지 않고 더한다고 읽지 말 것, 텐서는 불변처럼 새 값을 만든다 |
| `self.lm_head.weight = self.embed.tok.weight` | 두 필드가 **같은 객체**를 참조. 한쪽을 갱신하면 다른 쪽도 바뀐다 |
| `self.apply(self._init_weights)` | 모듈 트리를 재귀 순회하며 `Linear` 마다 초기화 콜백 실행 (visitor) |
| `model.eval()` / `model.train()` | 드롭아웃 같은 부품의 모드 플래그를 트리 전체에 설정 |
| `@dataclass GPTConfig` | 필드만 있는 설정 POJO. `GPTConfig(**dict)` 는 맵에서 생성자 호출 |

### 흔한 오해

- **"층을 많이 쌓으면 항상 좋다"**, 잔차와 LayerNorm 없이는 깊을수록 학습이 안 되고(기울기 소실), 있어도 데이터가 없으면 과적합만 빨라진다(9장 스케일링 표의 L4·C128).
- **"LayerNorm 은 BatchNorm 과 같다"**. BatchNorm 은 배치 방향으로 통계를 내서 문장 길이·배치 크기에 따라 결과가 달라진다. LayerNorm 은 벡터 하나 안에서만 계산해 그런 의존이 없다. 시퀀스 모델이 LayerNorm 을 쓰는 이유다.
- **"weight tying 은 파라미터를 아끼는 꼼수"**, 아끼기도 하지만 성능도 소폭 좋아진다. "토큰→벡터" 와 "벡터→토큰" 이 같은 기하를 쓰는 것이 자연스럽기 때문이다.
- **"GPT-2 와 우리 모델은 다른 것"**, 클래스 정의가 같다. `GPTConfig(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)` 이 GPT-2 small 이고, 노트북 §3.1 이 그 파라미터 수 1억 2,400만을 같은 코드로 센다.

## 6.7 실습: 노트북 06

`notebooks/06-transformer-gpt.ipynb` (전체 실행 약 4분):

1. LayerNorm 전후의 평균·표준편차, γ·β
2. 잔차 유무에 따른 32층 기울기 크기
3. `GPT` 조립, `(B, T) → (B, T, V)`, 초기 loss ≈ log V, 세 크기의 파라미터 표
4. 600 스텝 학습, 층·헤드별 어텐션 맵, 생성 미리보기

## 스스로 점검

1. x = [2, 4, 6, 8] 을 LayerNorm(γ=1, β=0) 에 넣으면? 손으로 계산하라.
2. 잔차 연결이 있을 때 32층 뒤에도 입력에 기울기가 도착하는 이유를 미분식 `d(x + f(x))/dx` 로 설명하라.
3. 블록 파라미터가 12·C² 인 이유를 어텐션 4C² 과 MLP 8C² 으로 나눠 설명하라. small-cpu 의 값은?
4. `(B, T, V)` 출력에서 자리 t 의 정답은 무엇이고, 그 예측이 컨닝이 아닌 이유는?
5. `model.eval()` 을 잊고 생성하면 어떤 일이 생기나?
6. GPT-2 small 과 small-cpu 의 차이를 숫자 다섯 개(V, T, L, H, C)로 말하라.

## 관련 코드

- `src/shllm/model.py`, `GPTConfig` · `MLP` · `Block` · `GPT` (`forward` · `n_params` · `attention_maps`)
- `src/shllm/attention.py`, `CausalSelfAttention` (5장), `src/shllm/embedding.py`, `GPTEmbedding` (3장)
- `tests/test_model.py`, 블록 shape 유지 · 파라미터 수 식 · 가중치 묶기 · 인과성 · 규칙 학습 · block_size 초과
