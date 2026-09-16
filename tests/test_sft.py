import torch

from shllm.model import GPT, GPTConfig
from shllm.sft import ANSWER, PROMPT, answer, build_examples, finetune
from shllm.tokenizer import BPETokenizer


def test_build_examples_format():
    ex = build_examples([("김유정", "봄봄")])
    assert len(ex) == 4
    assert ex[0].startswith(PROMPT) and ANSWER in ex[0] and ex[0].endswith("\n")
    assert "김유정" in ex[0] and "봄봄" in ex[0]


def test_finetune_teaches_format_and_answer():
    works = [("김유정", "봄봄"), ("현진건", "빈처"), ("이상", "날개")]
    examples = build_examples(works)
    tok = BPETokenizer.train("".join(examples) * 3, vocab_size=400)
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32))
    losses = finetune(model, tok, examples, steps=250, lr=3e-3, batch_size=6, log_every=0)
    assert losses[-1] < losses[0] * 0.3
    assert answer(model, tok, "봄봄의 작가는 누구인가?") == "김유정"
    assert answer(model, tok, "날개의 작가는 누구인가?") == "이상"
