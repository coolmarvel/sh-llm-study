import torch

from shllm.model import GPT, GPTConfig
from shllm.train import TrainConfig, Trainer, load_checkpoint, load_yaml_config, lr_at, read_log


def test_lr_schedule_shape():
    cfg = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=100)
    assert lr_at(0, cfg) == 0.1  # (0+1)/10 · 1.0
    assert lr_at(9, cfg) == 1.0
    assert abs(lr_at(55, cfg) - 0.55) < 1e-9  # 코사인 중간 = 평균
    assert abs(lr_at(100, cfg) - 0.1) < 1e-9
    assert lr_at(500, cfg) == 0.1


def test_trainer_reduces_loss_logs_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    seq = torch.tensor([i % 10 for i in range(500)])
    cfg = GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=32)
    tcfg = TrainConfig(
        run_name="t",
        batch_size=16,
        max_steps=120,
        lr=1e-2,
        min_lr=1e-3,
        warmup_steps=10,
        eval_every=40,
        eval_batches=4,
        ckpt_every=50,
    )
    runs, ckpts = tmp_path / "runs", tmp_path / "ckpt"
    trainer = Trainer(GPT(cfg), seq[:400], seq[400:], tcfg, runs_dir=runs, ckpt_dir=ckpts)
    hist = trainer.train(verbose=False)
    assert hist[0]["step"] == 0 and hist[-1]["step"] == 120
    assert hist[-1]["val_loss"] < hist[0]["val_loss"] * 0.5
    assert (runs / "t" / "config.yaml").exists()
    assert read_log("t", runs_dir=runs) == hist
    assert (ckpts / "t" / "ckpt.pt").exists() and (ckpts / "t" / "best.pt").exists()

    # 되살리기: 같은 예측, 같은 스텝
    model, ck = load_checkpoint("t", "ckpt.pt", ckpt_dir=ckpts)
    assert ck["step"] == 120 and ck["tokenizer"] == "bpe-8192.json"
    x = seq[None, :8]
    with torch.no_grad():
        trainer.model.eval()
        assert torch.allclose(model(x)[0], trainer.model(x)[0])

    # resume 후 이어서 학습
    t2 = Trainer(
        GPT(cfg),
        seq[:400],
        seq[400:],
        TrainConfig(**{**tcfg.__dict__, "max_steps": 140}),
        runs_dir=runs,
        ckpt_dir=ckpts,
    )
    assert t2.resume("ckpt.pt") and t2.step == 120
    t2.train(verbose=False)
    assert t2.step == 140


def test_load_yaml_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "model:\n  n_layer: 2\n  n_embd: 64\ntrain:\n  run_name: x\n  max_steps: 5\n  note: hi\n",
        "utf-8",
    )
    m, t = load_yaml_config(p)
    assert m.n_layer == 2 and m.n_embd == 64 and m.vocab_size == 8192
    assert t.run_name == "x" and t.max_steps == 5 and t.extra == {"note": "hi"}
