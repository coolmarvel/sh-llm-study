from shllm.tokenizer import BPETokenizer, CharTokenizer, _merge


def test_char_roundtrip(tmp_path):
    text = "옛날 옛적에 호랑이가 담배 피우던 시절에"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))
    assert max(ids) < tok.vocab_size

    tok.save(tmp_path / "char.json")
    assert CharTokenizer.load(tmp_path / "char.json").decode(ids) == text


def test_merge_helper():
    assert _merge([1, 2, 3, 1, 2, 1], (1, 2), 9) == [9, 3, 9, 1]
    assert _merge([1], (1, 2), 9) == [1]


def test_bpe_train_first_merge_and_roundtrip(tmp_path):
    text = "aaab aaab aaab ab"  # 가장 잦은 쌍은 'a','a'
    tok = BPETokenizer.train(text, vocab_size=256 + 3)
    assert tok.vocab_size == 259
    assert tok.merges[0] == (ord("a"), ord("a"))
    assert tok.vocab[256] == b"aa"
    assert tok.decode(tok.encode(text)) == text

    tok.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merges == tok.merges
    assert loaded.encode(text) == tok.encode(text)


def test_bpe_korean_roundtrip_and_unseen_chars():
    text = "옛날 옛적에 호랑이가 담배 피우던 시절에, 옛날 옛적에."
    tok = BPETokenizer.train(text, vocab_size=300)
    assert tok.decode(tok.encode(text)) == text
    # 학습에 없던 글자·이모지·영어도 바이트로 쪼개져 항상 인코딩된다 (CharTokenizer 는 KeyError)
    weird = "龍 🐯 hello 옛날"
    ids = tok.encode(weird)
    assert tok.decode(ids) == weird
    assert all(0 <= i < tok.vocab_size for i in ids)
    # 반복된 "옛날" 은 병합되어 바이트 수(6)보다 적은 토큰이 된다
    assert len(tok.encode("옛날")) < len("옛날".encode())


def test_bpe_truncated_is_prefix():
    text = "가나다 가나다 가나 가나다라 마바" * 5
    big = BPETokenizer.train(text, vocab_size=280)
    small = big.truncated(266)
    assert small.merges == big.merges[:10]
    assert small.vocab_size == 266
    assert small.decode(small.encode(text)) == text
    # 어휘가 작을수록 토큰 수는 같거나 많다
    assert len(small.encode(text)) >= len(big.encode(text))


def test_bpe_token_str_marks_partial_bytes():
    tok = BPETokenizer.train("가가가가", vocab_size=257)  # 첫 병합은 '가'(3바이트) 의 앞 두 바이트
    assert tok.token_str(256).startswith("<0x")
    assert tok.token_str(ord("a")) == "a"


def test_bpe_char_first_builds_whole_syllables():
    text = "옛날 옛적에 옛날 옛적에 호랑이 호랑이"
    tok = BPETokenizer.train(text, vocab_size=256 + 40, min_char_count=2)
    # 2번 이상 나온 글자는 전부 토큰 하나 (3바이트 → 병합 2번씩)
    for ch in "옛날적에호랑이":
        assert len(tok.encode(ch)) == 1, ch
    # 앞 2바이트가 같은 글자끼리는 중간 토큰을 공유하므로 병합 수 < 글자 수 × 2
    assert len(tok.merges) <= 40
    assert tok.decode(tok.encode(text)) == text


def test_bpe_rare_char_stays_bytes():
    text = "가가가가 龍"  # 龍 은 한 번 → min_char_count=2 에 걸려 조립되지 않는다
    # 예산 3 = '가' 조립 2 + 빈도 병합 1 ('가','가'). 예산이 남으면 빈도 병합이 결국 龍 의 바이트도 합친다
    tok = BPETokenizer.train(text, vocab_size=256 + 3, min_char_count=2)
    assert len(tok.encode("가")) == 1
    assert len(tok.encode("龍")) == 3  # 그대로 바이트 3개
    assert tok.decode(tok.encode(text)) == text
