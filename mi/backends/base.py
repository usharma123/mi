from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mi.core.schema import BehaviorSpec, LocalizationArtifact, TraceArtifact


class BackendUnavailable(RuntimeError):
    pass


class BackendCapabilityError(RuntimeError):
    pass


class TraceBackend(Protocol):
    name: str

    def trace(
        self,
        behavior: BehaviorSpec,
        activation_path: Path,
        *,
        run_id: str,
        top_k: int,
    ) -> TraceArtifact:
        ...

    def localize(
        self,
        behavior: BehaviorSpec,
        *,
        run_id: str,
        corrupt_prompt: str | None,
        methods: set[str],
        streams: set[str],
        controls: set[str],
        position: str,
        seed: int,
        top_k: int,
    ) -> LocalizationArtifact:
        ...


class NotImplementedTraceBackend:
    name = "not-implemented"

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device

    def trace(
        self,
        behavior: BehaviorSpec,
        activation_path: Path,
        *,
        run_id: str,
        top_k: int,
    ) -> TraceArtifact:
        raise BackendCapabilityError(f"{self.name} trace is not supported by this backend yet.")

    def localize(
        self,
        behavior: BehaviorSpec,
        *,
        run_id: str,
        corrupt_prompt: str | None,
        methods: set[str],
        streams: set[str],
        controls: set[str],
        position: str,
        seed: int,
        top_k: int,
    ) -> LocalizationArtifact:
        raise BackendCapabilityError(f"{self.name} localization is not supported by this backend yet.")

    def capabilities(self) -> dict[str, bool]:
        return {"trace": False, "localize": False, "features": False, "graph": False}
