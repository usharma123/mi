from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mi.core.schema import BehaviorSpec, LocalizationArtifact, TraceArtifact


class BackendUnavailable(RuntimeError):
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
        raise NotImplementedError(f"{self.name} backend is not implemented in mi v0.1.")

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
        raise NotImplementedError(f"{self.name} localization is not implemented yet.")
