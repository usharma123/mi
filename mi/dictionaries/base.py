from __future__ import annotations

from typing import Protocol

from pydantic import Field

from mi.core.schema import FeatureRef, MIModel


class FeatureLookupResult(MIModel):
    feature: FeatureRef
    activation: float | None = None
    examples: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class FeatureDictionary(Protocol):
    dictionary_id: str

    def top_features(self, activation, *, top_k: int) -> list[FeatureLookupResult]:
        ...


class NotImplementedFeatureDictionary:
    dictionary_id = "not-implemented"

    def top_features(self, activation, *, top_k: int) -> list[FeatureLookupResult]:
        raise NotImplementedError(
            f"{self.dictionary_id} dictionary adapter is not implemented in mi v0.1."
        )
