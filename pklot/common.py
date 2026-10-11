"""Paths and small helpers shared by the training and inference scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def pick_device():
    """Best available torch device: CUDA GPU, then Apple MPS, then CPU.

    Returned in the form Ultralytics accepts (0, "mps" or "cpu").
    """
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
