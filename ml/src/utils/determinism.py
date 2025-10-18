from typing import Optional

import torch


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = torch.seed() % (2**31 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def configure_determinism(enabled: bool = True) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
