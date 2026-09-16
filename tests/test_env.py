import torch

from shllm import __version__
from shllm.config import REPO_ROOT, setup_cpu


def test_version():
    assert __version__


def test_cpu_setup():
    device = setup_cpu(num_threads=2)
    assert device.type == "cpu"
    assert torch.get_num_threads() == 2
    x = torch.randn(4, 4)
    assert (x @ x.T).shape == (4, 4)


def test_repo_root():
    assert (REPO_ROOT / "pyproject.toml").exists()
