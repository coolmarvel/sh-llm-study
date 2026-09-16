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
| 토크나이저 | `tokenizer.py` `BPETokenizer` | 바이트 BPE + 글자 조립(`char_first`), `data/tokenizers/bpe-8192.json` |
| 모델 | `model.py` `GPT` | Pre-LN 블록 × L, weight tying, `attention_maps()` |
| 학습 | `train.py` `Trainer` | warmup+cosine, AdamW(decay 는 행렬만), clip 1.0, JSONL 로그, best/ckpt |
| 긴 학습 | `scripts/train.py --config configs/*.yaml` | 기본 `--threads` = 물리 코어. `--resume` |
| 생성 | `generate.py` | 반복 억제 → temperature → top-k → top-p |
| 평가 | `eval.py` | `evaluate_loss`(시드 고정) · `bits_per_char` · `scaling_experiment` |
| SFT 시연 | `sft.py` | 답 부분만 loss, `WORKS` 로 질문-답 |

## 파일 규약 (data/ = /mnt/d/sh-llm-data)

```
tokenizers/bpe-8192.json
runs/<run>/log.jsonl        {"step","train_loss","val_loss","lr","elapsed"}
runs/<run>/config.yaml      {model: GPTConfig, train: TrainConfig}
runs/<run>.out              nohup 표준 출력 (scripts/train.py)
checkpoints/<run>/ckpt.pt   {"model","optimizer","step","best_val","model_cfg","train_cfg","tokenizer"}
checkpoints/<run>/best.pt   같은 형식, val 최저 시점
```

불변: **토크나이저 파일을 바꾸면 모든 체크포인트가 무효**다. 체크포인트에 토크나이저 이름을 기록하는 이유.

## 함정 (근거 1줄)

- `torch.set_num_threads(16)` 은 8 보다 16배 느리다 — `setup_cpu()` 기본은 물리 코어 (`config.PHYSICAL_CORES`, 2026-09-16 측정).
- 코퍼스 49만 토큰 → 6.8M 모델은 약 1,000 스텝(≈ 8 에폭)부터 val 이 정체·상승. `best.pt` 를 쓴다.
- 검증 loss 를 전체 val 로 한 번에 계산하면 (N, V) softmax 가 GB 단위 — 배치 평균(`evaluate_loss`)으로.

## 관련 코드

- `src/shllm/{tokenizer,model,attention,embedding,train,generate,eval,sft}.py`, `scripts/train.py`, `configs/{tiny-notebook,small-cpu}.yaml`
- 테스트: `tests/test_{tokenizer,model,attention,embedding,train,generate,eval,sft}.py`
