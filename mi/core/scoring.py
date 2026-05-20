from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalTriangulationScores:
    association: float | None = None
    attribution: float | None = None
    necessity: float | None = None
    sufficiency: float | None = None
    robustness: float | None = None
    specificity: float | None = None
    dictionary: float | None = None
    label_consistency: float | None = None

    def mean_available(self) -> float | None:
        values = [value for value in self.__dict__.values() if value is not None]
        if not values:
            return None
        return sum(values) / len(values)
