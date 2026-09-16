"""shllm — 밑바닥부터 만드는 소형 한글 GPT (학습용 패키지).

장별 모듈:
    config     경로·CPU 설정 (SSOT)          data       코퍼스·작품 목록·get_batch
    tokenizer  (1~2장) CharTokenizer · BPETokenizer
    bigram     (1장)   세어서 나누는 n-gram — 확률·생성·loss 의 첫 구현
    embedding  (3장)   토큰·위치 임베딩, 세어서 만든 벡터(PPMI+SVD)
    autograd   (4장)   스칼라 자동미분 Value      numpy_lm (4장) 손으로 쓴 역전파    mlp (4장) MLP 언어모델
    attention  (5장)   셀프 어텐션 (단일·멀티헤드, 인과 마스크)
    model      (6장)   Transformer 블록 → GPT
    train      (7장)   Trainer · 체크포인트 · JSONL 로그
    generate   (8장)   temperature · top-k · top-p · 반복 억제
    eval       (9장)   perplexity · bits/char · 스케일링 실험
    sft        (10장)  지시 파인튜닝 미니 시연
"""

__version__ = "0.1.4"
