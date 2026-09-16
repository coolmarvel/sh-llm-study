---
title: 0장 — 오리엔테이션: LLM 의 큰 그림과 환경 준비
created: 2026-09-16
updated: 2026-09-16
domain: book
---

# 0장 — 오리엔테이션

이 교재는 **자바 개발자가 LLM 을 밑바닥부터 직접 만들어 보는 여정**이다. 각 장은 "개념 → 코드 → 실습"
순서이고, 노트북(`notebooks/00-orientation.ipynb`)을 함께 열어 셀을 실행하며 읽는다.

## 0.1 LLM 은 결국 무엇을 하는 기계인가

한 문장으로: **"지금까지 나온 글을 보고, 다음에 올 토큰 하나의 확률을 계산하는 함수"** 다.

```
입력: "옛날 옛적에 호랑이가"      →  모델  →  다음 토큰 확률: {" 담배": 0.31, " 살았": 0.22, " 나타": 0.09, ...}
```

이 함수를 한 번 돌려 토큰 하나를 고르고, 그 토큰을 입력 끝에 붙여 다시 돌리는 것을 반복하면 문장이
"생성"된다. ChatGPT 도 이 반복이 전부다. 나머지(대화 형식, 지시 따르기, 안전성)는 이 함수를 **어떤
데이터로 학습시켰는가**의 차이일 뿐이다.

그래서 우리가 만들 것은 정확히 이 함수 하나와, 그것을 데이터로 학습시키는 루프다.

## 0.2 여정 지도 — 10개 장이 하나로 이어진다

| 장 | 질문 | 만드는 것 |
|---|---|---|
| 1 | 글자를 어떻게 숫자로 바꾸나? 확률표만으로 글이 써지나? | 문자 토크나이저, 바이그램 확률표 |
| 2 | 토큰 단위를 글자보다 크게 잡으면 뭐가 좋은가? | BPE 토크나이저 |
| 3 | 정수 id 를 왜 벡터로 바꾸나? | 토큰·위치 임베딩 |
| 4 | 학습이란 정확히 무엇을 하는 것인가? | 경사하강, autograd, MLP 언어모델 |
| 5 | 앞 단어들 중 어디를 봐야 하는지 모델이 어떻게 아나? | 셀프 어텐션 |
| 6 | 어텐션을 어떻게 쌓아 GPT 가 되나? | Transformer 블록, GPT |
| 7 | 이 PC 에서 몇 시간 안에 학습시키려면? | 학습 루프, 체크포인트, 손실 곡선 |
| 8 | 확률에서 문장을 어떻게 뽑나? | temperature, top-k, top-p |
| 9 | 잘 됐는지 어떻게 아나? 크게 만들면 좋아지나? | perplexity, 스케일링 실험 |
| 10 | 여기서 ChatGPT 까지는 무엇이 더 필요한가? | 파인튜닝·RLHF·RAG 개념 지도 |

각 장의 코드는 `src/shllm/` 패키지에 쌓인다. 7~8장이 끝나면 `shllm` 만으로 학습·생성이 되는
"내 GPT" 한 벌이 완성된다.

## 0.3 자바 개발자를 위한 Python·PyTorch 최소 지식

Python 은 자바보다 문법이 느슨하지만, 이 프로젝트에서 쓰는 범위는 좁다. 아래만 알면 시작할 수 있다.

| 자바 | Python | 비고 |
|---|---|---|
| `List<Integer> a = new ArrayList<>();` | `a: list[int] = []` | 타입 힌트는 문서일 뿐, 강제하지 않는다 |
| `Map<String,Integer>` | `dict[str, int]` | `{"a": 1}` |
| `for (int i=0; i<n; i++)` | `for i in range(n):` | 블록은 들여쓰기로 |
| `class Foo { Foo(int x) {...} }` | `class Foo:` + `def __init__(self, x):` | `self` 를 명시 |
| `interface` / 추상 클래스 | `torch.nn.Module` 상속 + `forward()` 구현 | 모델은 전부 이 패턴 |
| `import java.util.*` | `import torch` / `from shllm.tokenizer import CharTokenizer` | |
| Maven/Gradle | **uv** (`pyproject.toml` + `uv.lock`) | `uv sync` = 의존성 복원 |
| JUnit | **pytest** (`tests/test_*.py`) | `uv run pytest` |
| `double[][]` | `torch.Tensor` | 차원이 몇 개든 하나의 타입 |

**텐서(Tensor)** 는 "다차원 배열 + 자동미분"이다. `double[][]` 와 다른 점은 셋뿐이다.

1. 연산이 배열 전체에 한 번에 적용된다 (`x * 2`, `x @ w` — 루프를 쓰지 않는다).
2. 모양(`shape`)이 항상 붙어 다닌다. LLM 코드의 버그 대부분은 shape 불일치다. 주석에 shape 를 적는 습관을 들인다.
   예: `x: (B, T, C)` = 배치 B개, 토큰 T개, 채널 C개.
3. `requires_grad=True` 인 텐서는 연산 이력을 기억해 두었다가 `.backward()` 로 미분값을 채운다 (4장).

**PyTorch 모델의 뼈대**는 언제나 이 형태다. 6장까지의 모든 모듈이 이 틀을 따른다.

```python
import torch.nn as nn

class Foo(nn.Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)   # 학습되는 파라미터를 가진 부품

    def forward(self, x):                      # x: (B, n_in)
        return self.linear(x)                  # → (B, n_out)
```

## 0.4 환경

| 항목 | 값 |
|---|---|
| OS | WSL2 Ubuntu, Python 3.12 |
| 하드웨어 | CPU 16 스레드(물리 8코어), RAM 7.6GB, **GPU 없음** |
| PyTorch | CPU 빌드 (`torch==2.14+cpu`) |
| 코드 | `~/sh-llm-study` (WSL 파일시스템) |
| 데이터 | `data/` → `/mnt/d/sh-llm-data` (D 드라이브, 윈도우 탐색기에서 `D:\sh-llm-data`) |

CPU 로 학습한다는 것은 GPU 대비 수십 배 느리다는 뜻이다. 그래서 이 교재는 모델 크기를 장마다 명시하고,
**노트북 셀은 10분 이내, 긴 학습은 `scripts/train.py` 로 분리**한다. 원리를 배우는 데는 수백만
파라미터면 충분하다 — GPT-2 small(1.2억)도 구조는 같다.

## 0.5 코퍼스 — 한국 근대 단편·장편 20편

`scripts/download_corpus.py` 가 위키문헌(ko.wikisource.org)에서 저작권이 만료된 작품 20편을 받아
`data/corpus/works/` 에 두고, 합본 `data/corpus/korean-classics.txt` 를 만든다. 합본은 약 114만 자 — 영어 교재들이 쓰는
Tiny Shakespeare(110만 자)와 같은 급이다.

| 작가 | 작품 |
|---|---|
| 김유정 | 봄봄 · 동백꽃 · 소낙비 · 만무방 |
| 현진건 | 운수 좋은 날 · 빈처 · B사감과 러브레터 · 술 권하는 사회 |
| 이상 | 날개 |
| 이효석 | 메밀꽃 필 무렵 |
| 나도향 | 벙어리 삼룡이 · 물레방아 |
| 김동인 | 감자 · 배따라기 · 광염 소나타 |
| 채만식 | 레디메이드 인생 · 치숙 · 탁류 · 태평천하 |
| 이광수 | 무정 |

1930년대 문체라 결과물도 그 말투를 닮게 된다. 이것이 "학습 데이터가 모델을 결정한다"의 첫 체험이다.

## 0.6 실습 — 노트북 00

`notebooks/00-orientation.ipynb` 에서 확인하는 것:

1. PyTorch 가 CPU 스레드를 몇 개 쓰는지, 행렬곱 속도가 얼마인지
2. 코퍼스를 읽어 글자 수·고유 글자 수·가장 흔한 글자를 세기
3. 텐서 shape 감각 — `(B, T, C)` 를 만들고 바꿔 보기

## 관련 코드

- `src/shllm/config.py` — 경로(`DATA_DIR`…)와 `setup_cpu()`
- `src/shllm/data.py` — `load_corpus()` · `list_work_files()`
- `scripts/download_corpus.py` — 코퍼스 수집·정리(`drop_boilerplate`)
