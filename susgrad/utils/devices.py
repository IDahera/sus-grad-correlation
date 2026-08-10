"""Device selection: prefer Apple MPS, then CUDA, then CPU."""

import torch


def get_best_device() -> torch.device:
    """Return the best available torch device.

    Preference order: MPS (Apple Silicon) -> CUDA (NVIDIA) -> CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_best_device()
