from shllm.tokenizer import CharTokenizer


def test_char_roundtrip(tmp_path):
    text = "옛날 옛적에 호랑이가 담배 피우던 시절에"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))
    assert max(ids) < tok.vocab_size

    tok.save(tmp_path / "char.json")
    assert CharTokenizer.load(tmp_path / "char.json").decode(ids) == text
