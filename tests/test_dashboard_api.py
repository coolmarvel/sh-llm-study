import json

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from shllm.model import GPT, GPTConfig
from shllm.tokenizer import BPETokenizer


@pytest.fixture
def client(tmp_path, monkeypatch):
    # 임시 data 디렉토리에 가짜 run 하나 + 체크포인트 + 토크나이저를 만든다
    runs, ckpts, toks = tmp_path / "runs", tmp_path / "checkpoints", tmp_path / "tokenizers"
    (runs / "demo").mkdir(parents=True)
    (ckpts / "demo").mkdir(parents=True)
    toks.mkdir()
    tok = BPETokenizer.train("옛날 옛적에 호랑이가 담배 피우던 시절에 " * 5, vocab_size=300)
    tok.save(toks / "t.json")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=16)
    model = GPT(cfg)
    torch.save(
        {"model": model.state_dict(), "model_cfg": cfg.__dict__, "step": 3, "tokenizer": "t.json"},
        ckpts / "demo" / "best.pt",
    )
    log = [
        {"step": 0, "train_loss": 5.0, "val_loss": 5.1, "lr": 1e-3, "elapsed": 1},
        {"step": 10, "train_loss": 4.0, "val_loss": 4.2, "lr": 1e-3, "elapsed": 9},
    ]
    (runs / "demo" / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    (runs / "demo" / "config.yaml").write_text(
        yaml.safe_dump({"model": cfg.__dict__, "train": {"run_name": "demo"}})
    )

    import shllm.config as c
    import shllm.train as t
    from dashboard.api import main

    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "CHECKPOINT_DIR", ckpts)
    monkeypatch.setattr(main, "TOKENIZER_DIR", toks)
    monkeypatch.setattr(t, "RUNS_DIR", runs)
    monkeypatch.setattr(t, "CHECKPOINT_DIR", ckpts)
    monkeypatch.setattr(c, "RUNS_DIR", runs)
    main._load.cache_clear()
    # load_checkpoint / read_log 는 기본 인자를 import 시점에 묶으므로 명시적으로 넘기는 래퍼로 교체
    monkeypatch.setattr(
        main,
        "load_checkpoint",
        lambda name, f="best.pt": t.load_checkpoint(name, f, ckpt_dir=ckpts),
    )
    monkeypatch.setattr(main, "read_log", lambda name: t.read_log(name, runs_dir=runs))
    return TestClient(main.app)


def test_runs_and_log(client):
    runs = client.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["name"] == "demo"
    assert runs[0]["best_val"] == 4.2 and runs[0]["best_step"] == 10 and runs[0]["has_checkpoint"]
    assert runs[0]["n_params"] > 0
    log = client.get("/api/runs/demo/log").json()
    assert [r["step"] for r in log] == [0, 10]
    assert client.get("/api/runs/nope/log").status_code == 404


def test_attention_and_generate(client):
    a = client.post("/api/attention", json={"run": "demo", "text": "옛날 옛적에"}).json()
    assert a["n_layer"] == 2 and a["n_head"] == 2
    T = len(a["tokens"])
    assert len(a["maps"]) == 2 and len(a["maps"][0][0]) == T
    assert abs(sum(a["maps"][0][0][-1]) - 1) < 1e-4  # 마지막 행의 합 1

    g = client.post(
        "/api/generate", json={"run": "demo", "prompt": "옛날", "max_new_tokens": 5, "top_n": 3}
    ).json()
    assert len(g["steps"]) == 5 and len(g["steps"][0]["candidates"]) == 3
    assert g["text"].startswith("옛날")
    assert all(0 <= s["prob"] <= 1 for s in g["steps"])
    # 같은 seed 면 같은 결과
    g2 = client.post(
        "/api/generate", json={"run": "demo", "prompt": "옛날", "max_new_tokens": 5, "top_n": 3}
    ).json()
    assert g2["text"] == g["text"]
    tk = client.get("/api/tokenize", params={"run": "demo", "text": "옛날 옛적에"}).json()
    assert len(tk["ids"]) == len(tk["tokens"]) > 0


def test_book_and_notebooks(client, tmp_path, monkeypatch):
    from dashboard.api import main

    book = tmp_path / "book"
    book.mkdir()
    (book / "01-a.md").write_text(
        "---\ntitle: 1장 테스트\n---\n# 제목\n\n본문 `code`\n\n```python\nx = 1\n```\n", "utf-8"
    )
    nbs, ex = tmp_path / "nbs", tmp_path / "ex"
    nbs.mkdir()
    ex.mkdir()
    import nbformat

    nb = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 노트북 제목"),
            nbformat.v4.new_code_cell("print(1)"),
        ]
    )
    nbformat.write(nb, nbs / "01-a.ipynb")
    nb.cells[1].outputs = [nbformat.v4.new_output("stream", name="stdout", text="1\n")]
    nbformat.write(nb, ex / "01-a.ipynb")
    monkeypatch.setattr(main, "BOOK_DIR", book)
    monkeypatch.setattr(main, "NOTEBOOK_DIR", nbs)
    monkeypatch.setattr(main, "EXECUTED_DIR", ex)
    assert client.get("/api/book").json() == [{"name": "01-a", "title": "1장 테스트"}]
    ch = client.get("/api/book/01-a").json()
    assert "<h1" in ch["html"] and "codehilite" in ch["html"] and ch["css"]
    assert client.get("/api/book/zz").status_code == 404
    lst = client.get("/api/notebooks").json()
    assert (
        lst[0]["title"] == "노트북 제목"
        and lst[0]["executed"]
        and lst[0]["lab_url"].endswith("01-a.ipynb")
    )
    html = client.get("/api/notebooks/01-a/html").text
    assert "노트북 제목" in html and ">1\n<" in html or "1" in html
    assert client.get("/api/notebooks/nope/html").status_code == 404
    # 셀 읽기·저장 (저장은 출력을 버리고 순번 id 로)
    cells = client.get("/api/notebooks/01-a/cells").json()
    assert [c["cell_type"] for c in cells["cells"]] == ["markdown", "code"]
    assert cells["cells"][1]["outputs"][0]["text"] == "1\n"  # 실행본 출력이 초기값
    assert client.put(
        "/api/notebooks/01-a/cells",
        json={"cells": [{"id": "0", "cell_type": "code", "source": "x = 2"}]},
    ).json() == {"saved": 1}
    assert client.get("/api/notebooks/01-a/cells").json()["cells"][0]["source"] == "x = 2"


def test_generate_stream(client):
    import json as _json

    r = client.post(
        "/api/generate/stream",
        json={"run": "demo", "prompt": "옛날", "max_new_tokens": 4, "top_n": 2},
    )
    assert r.status_code == 200
    lines = [_json.loads(x) for x in r.text.strip().split("\n")]
    assert "prompt_tokens" in lines[0] and "text" in lines[-1]
    assert len(lines) == 6 and all("token" in x for x in lines[1:-1])


def test_kernel_execute_stream_and_source(client, tmp_path, monkeypatch):
    import json as _json

    from dashboard.api import main
    from dashboard.api.kernels import KernelPool

    monkeypatch.setattr(
        main, "KERNELS", KernelPool(cwd=tmp_path, pythonpath=main.REPO_ROOT / "src")
    )
    try:
        r = client.post(
            "/api/notebooks/t/execute", json={"code": "import shllm\nprint('hi', 1 + 1)\n3 * 3"}
        )
        outs = [_json.loads(x) for x in r.text.strip().split("\n")]
        kinds = [o["output_type"] for o in outs]
        assert "stream" in kinds and "execute_result" in kinds and outs[-1]["state"] == "idle"
        assert outs[-1]["execution_count"] == 1
        assert any(o.get("text") == "hi 2\n" for o in outs)
        assert any(o.get("data", {}).get("text/plain") == "9" for o in outs)
        # 상태 유지: 앞 셀의 변수가 다음 셀에 보인다
        r = client.post("/api/notebooks/t/execute", json={"code": "y = 5\ny"})
        r = client.post("/api/notebooks/t/execute", json={"code": "y + 1"})
        assert any(
            o.get("data", {}).get("text/plain") == "6"
            for o in (_json.loads(x) for x in r.text.strip().split("\n"))
        )
        # 에러
        r = client.post("/api/notebooks/t/execute", json={"code": "1/0"})
        assert any(
            o.get("ename") == "ZeroDivisionError"
            for o in (_json.loads(x) for x in r.text.strip().split("\n"))
        )
        assert client.get("/api/kernels").json() == ["t"]
        assert client.post("/api/notebooks/t/kernel/restart").json()["alive"]
    finally:
        main.KERNELS.shutdown()
    src = client.get("/api/source", params={"path": "src/shllm/config.py"}).json()
    assert "setup_cpu" in src["html"] and src["lines"] > 10
    assert client.get("/api/source", params={"path": "../etc/passwd"}).status_code == 400
