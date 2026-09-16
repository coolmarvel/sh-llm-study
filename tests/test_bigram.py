import math

import torch

from shllm.bigram import BigramModel, NGramModel
from shllm.tokenizer import CharTokenizer

TEXT = "가나다가나다가나다가나라"  # "가"→"나", "나"→"다" 는 항상, "다"→"가" 3번, "나"→"라" 1번


def _ids():
    tok = CharTokenizer.from_text(TEXT)
    return tok, tok.encode(TEXT)


def test_bigram_counts_and_probs():
    tok, ids = _ids()
    m = BigramModel(tok.vocab_size, smoothing=0.0).fit(ids)
    ga, na, da, ra = (tok.stoi[c] for c in "가나다라")
    assert m.counts[ga, na] == 4
    assert m.counts[na, da] == 3
    assert m.counts[na, ra] == 1
    assert m.counts[da, ga] == 3
    assert m.counts.sum() == len(ids) - 1
    assert torch.allclose(m.probs.sum(dim=1), torch.ones(tok.vocab_size))
    assert math.isclose(float(m.probs[na, da]), 0.75)


def test_bigram_generate_and_loss():
    tok, ids = _ids()
    m = BigramModel(tok.vocab_size, smoothing=0.0).fit(ids)
    g = torch.Generator().manual_seed(0)
    out = m.generate(tok.stoi["가"], 20, generator=g)
    assert len(out) == 21
    assert all(0 <= i < tok.vocab_size for i in out)
    # 확률 0 인 쌍은 절대 안 나온다: "가" 다음은 언제나 "나"
    text = tok.decode(out)
    assert "가나" in text and "가다" not in text and "가라" not in text
    # 같은 시드면 같은 결과
    g2 = torch.Generator().manual_seed(0)
    assert m.generate(tok.stoi["가"], 20, generator=g2) == out
    # 학습 텍스트의 loss 는 "아무 정보 없음"(log V) 보다 낮다
    assert m.loss(ids) < math.log(tok.vocab_size)
    assert math.isclose(m.perplexity(ids), math.exp(m.loss(ids)))


def test_smoothing_prevents_infinite_loss():
    tok, ids = _ids()
    unseen = tok.encode("가라")  # "가"→"라" 는 학습에 없음
    assert math.isinf(BigramModel(tok.vocab_size, smoothing=0.0).fit(ids).loss(unseen))
    assert math.isfinite(BigramModel(tok.vocab_size, smoothing=1.0).fit(ids).loss(unseen))


def test_ngram_probs_sum_to_one_and_fall_back_to_shorter_context():
    tok, ids = _ids()
    m = NGramModel(3, tok.vocab_size, smoothing=1.0).fit(ids)
    assert (
        m.n_contexts == 3
    )  # 등장한 2글자 문맥: 가나 · 나다 · 다가 (나라 는 맨 끝이라 다음 글자가 없다)
    for ctx in ("가나", "라라", "나"):
        p = m.next_token_probs(tok.encode(ctx))
        assert math.isclose(float(p.sum()), 1.0, rel_tol=1e-5)
    # 처음 보는 문맥 "라라" → 직전 한 글자 "라"도 앞자리에 없음 → 유니그램(+균등) 분포로 후퇴
    assert torch.allclose(m.next_token_probs(tok.encode("라라")), m.next_token_probs([]))
    # 본 문맥 "가나" 다음은 항상 "다"(3번) 였으니 압도적
    assert float(m.next_token_probs(tok.encode("가나"))[tok.stoi["다"]]) > 0.7
    # 벡터 계산과 스칼라 계산이 같은 식인지
    p = m.next_token_probs(tok.encode("가나"))
    assert math.isclose(
        float(p[tok.stoi["다"]]), m._prob(tok.encode("가나"), 2, tok.stoi["다"]), rel_tol=1e-5
    )


def test_unigram_equals_frequency():
    tok, ids = _ids()
    m = NGramModel(1, tok.vocab_size, smoothing=0.0).fit(ids)
    p = m.next_token_probs([])
    assert math.isclose(float(p[tok.stoi["가"]]), TEXT.count("가") / len(TEXT), rel_tol=1e-5)


def test_ngram_longer_context_fits_better():
    # "나" 다음은 "다" 또는 "마", 직전 글자만 보면 반반이지만, 두 글자를 보면("가나"/"라나") 확정된다
    text = "가나다라나마" * 10
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    losses = [NGramModel(n, tok.vocab_size, smoothing=0.1).fit(ids).loss(ids) for n in (1, 2, 3)]
    assert losses[0] > losses[1] > losses[2]
    assert losses[2] < 0.05  # 트라이그램은 거의 완벽히 예측한다
    g = torch.Generator().manual_seed(0)
    out = tok.decode(
        NGramModel(3, tok.vocab_size, smoothing=0.1).fit(ids).generate(tok.encode("가나"), 30, g)
    )
    assert out.startswith("가나다라나마가나다라나마")
