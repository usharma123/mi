from __future__ import annotations

import numpy as np


def zero_ablate(vector: np.ndarray) -> np.ndarray:
    return np.zeros_like(vector)


def ablation_delta(metric_before: float, metric_after: float) -> float:
    return float(metric_after - metric_before)
