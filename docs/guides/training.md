---
title: 학습·생성·평가 파이프라인
created: 2026-09-16
updated: 2026-09-16
domain: model
---

# 학습·생성·평가 파이프라인

## 개요

`src/shllm/` 패키지로 토크나이저 → 모델 → 학습 → 생성 → 평가가 한 벌로 돈다. 장별 설명은 `docs/book/`, 여기는 **현재 구현된 동작과 파일 규약**만.

## 동작 방식

```
corpus/korean-classics.txt ─BPETokenizer(bpe-8192.json)─▶ ids (542K 토큰)
   └ 90/10 분할 ─get_batch─▶ (B, T) / (B, T)
GPT(GPTConfig) ─Trainer(TrainConfig)─▶ runs/<run>/log.jsonl · checkpoints/<run>/{ckpt,best}.pt
best.pt ─load_checkpoint─▶ generate(...) / evaluate_loss(...) / 대시보드
```

| 단계 | 파일 | 핵심 |
|---|---|---|
| 토크나이저 | `tokenizer.py` `BPETokenizer` · `scripts/train_tokenizer.py` | 바이트 BPE + 글자 조립(`char_first`). `bpe-8192.json`(근대문학) · `bpe-8192-mixed.json`(근대문학+위키, ADR-0003) |
| 모델 | `model.py` `GPT` | Pre-LN 블록 × L, weight tying, `attention_maps()` |
| 학습 | `train.py` `Trainer` | warmup+cosine, AdamW(decay 는 행렬만), clip 1.0, JSONL 로그, best/ckpt |
| 긴 학습 | `scripts/train.py --config configs/*.yaml` | 기본 `--threads` = 물리 코어. `--resume`. 설정의 `corpus`·`tokenizer` 로 데이터를 고른다 |
| 코퍼스 | `scripts/download_corpus.py`(근대문학 20편) · `scripts/download_wiki.py`(위키 알찬·좋은 글 + 링크 이웃, 약 1,000만 자) | `korean-classics.txt` · `korean-wiki.txt` · `korean-mixed.txt` |
| 생성 | `generate.py` | 반복 억제 → temperature → top-k → top-p |
| 평가 | `eval.py` | `evaluate_loss`(시드 고정) · `bits_per_char` · `scaling_experiment` |
| SFT 시연 | `sft.py` | 답 부분만 loss, `WORKS` 로 질문-답 |

## 파일 규약 (data/ = /mnt/d/sh-llm-data)

```
corpus/korean-classics.txt         근대문학 20편 합본 (1~9장 노트북)
corpus/korean-wiki.txt             위키 673편 합본, corpus/WIKI-LICENSE.md (CC BY-SA 출처)
corpus/korean-mixed.txt            둘을 이어 붙인 것 (configs/mixed-cpu.yaml)
tokenizers/bpe-8192.json           근대문학용 · tokenizers/bpe-8192-mixed.json 혼합용
runs/<run>/log.jsonl        {"step","train_loss","val_loss","lr","elapsed"}
runs/<run>/config.yaml      {model: GPTConfig, train: TrainConfig}
runs/<run>.out              nohup 표준 출력 (scripts/train.py)
checkpoints/<run>/ckpt.pt   {"model","optimizer","step","best_val","model_cfg","train_cfg","tokenizer"}
checkpoints/<run>/best.pt   같은 형식, val 최저 시점
```

불변: **토크나이저 파일을 바꾸면 모든 체크포인트가 무효**다. 체크포인트에 토크나이저 이름을 기록하는 이유.

## 내 데이터로 학습하기

1. UTF-8 텍스트 하나를 `data/corpus/<이름>.txt` 에 둔다 (`D:\sh-llm-data\corpus\`). 저작권은 본인 책임, 이 프로젝트는 퍼블릭 도메인·CC BY-SA 만 쓴다.
2. `uv run python scripts/train_tokenizer.py --corpus <이름> --out bpe-8192-<이름>.json`
3. `configs/small-cpu.yaml` 을 복사해 `run_name` · `corpus` · `tokenizer` 를 바꾸고 `uv run python scripts/train.py --config configs/<이름>.yaml`
4. 결과는 `data/checkpoints/<run_name>/best.pt`. 대시보드(8082)에 자동으로 나타난다. 중단 후 `--resume`.
5. 이미 학습된 모델 위에 이어 학습(파인튜닝)하려면 **같은 토크나이저**여야 한다. 10장 `sft.finetune` 이 예.

용량: 체크포인트 하나 ≈ 파라미터 × 12 바이트 (모델 4 + AdamW 8). 6.8M 이면 79MB, run 마다 best·ckpt 두 개.

## 함정 (근거 1줄)

- `torch.set_num_threads(16)` 은 8 보다 16배 느리다. `setup_cpu()` 기본은 물리 코어 (`config.PHYSICAL_CORES`, 2026-09-16 측정).
- 코퍼스 49만 토큰 → 6.8M 모델은 약 2,100 스텝부터 val 정체 (best 5.485). `best.pt` 를 쓴다. 데이터 확장(ADR-0003)이 근본 해법.
- 검증 loss 를 전체 val 로 한 번에 계산하면 (N, V) softmax 가 GB 단위, 배치 평균(`evaluate_loss`)으로.

## 관련 코드

- `src/shllm/{tokenizer,model,attention,embedding,train,generate,eval,sft}.py`, `scripts/train.py`, `configs/{tiny-notebook,small-cpu}.yaml`
- 테스트: `tests/test_{tokenizer,model,attention,embedding,train,generate,eval,sft}.py`
