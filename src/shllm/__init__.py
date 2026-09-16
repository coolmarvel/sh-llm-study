"""shllm — 밑바닥부터 만드는 소형 한글 GPT (학습용 패키지).

챕터가 진행되면서 모듈이 하나씩 채워진다:
    tokenizer  (1~2장)  텍스트 ↔ 정수 토큰
    bigram     (1장)    세어서 나누는 n-gram 언어모델 — 확률·생성·loss 의 첫 구현
    embedding  (3장)    토큰 → 벡터
    attention  (5장)    셀프 어텐션
    model      (6장)    Transformer 블록 → GPT
    train      (7장)    학습 루프·체크포인트
    generate   (8장)    텍스트 생성
"""

__version__ = "0.1.3"
