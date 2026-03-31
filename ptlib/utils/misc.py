import numpy as np

import torch
from collections.abc import Mapping, Sequence


def random_binary_mask(
    N: int, k: int, *, rng: np.random.Generator | None = None
) -> np.ndarray:
    if N < 0:
        raise ValueError("N must be >= 0")
    if k < 0 or k > N:
        raise ValueError("k must satisfy 0 <= k <= N")

    rng = np.random.default_rng() if rng is None else rng
    mask = np.zeros(N, dtype=bool)
    if k == 0:
        return mask

    idx = rng.choice(N, size=k, replace=False)
    mask[idx] = True
    return mask


def torch_to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    # dict-like
    if isinstance(obj, Mapping):
        return obj.__class__({k: torch_to_numpy(v) for k, v in obj.items()})

    # list / tuple
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return obj.__class__(torch_to_numpy(v) for v in obj)

    return obj
