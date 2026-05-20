from __future__ import annotations

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def rank_of(scores: np.ndarray, index: int) -> int:
    return int(np.sum(scores > scores[index]) + 1)


def normalize_effect(delta: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(delta / denominator)
