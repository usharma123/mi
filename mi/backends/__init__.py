from __future__ import annotations

from collections.abc import Callable

from mi.backends.base import TraceBackend
from mi.backends.circuit_tracer_backend import CircuitTracerBackend
from mi.backends.hf_hooks_backend import HFHooksBackend
from mi.backends.nnsight_backend import NNsightBackend
from mi.backends.transformer_lens_backend import TransformerLensBackend


def get_backend(name: str) -> Callable[[str, str | None], TraceBackend]:
    normalized = name.lower().replace("_", "-")
    if normalized in {"transformer-lens", "transformerlens", "tl"}:
        return lambda model_name, device=None: TransformerLensBackend(model_name, device=device)
    if normalized == "nnsight":
        return lambda model_name, device=None: NNsightBackend(model_name, device=device)
    if normalized in {"hf-hooks", "hf"}:
        return lambda model_name, device=None: HFHooksBackend(model_name, device=device)
    if normalized in {"circuit-tracer", "circuittracer"}:
        return lambda model_name, device=None: CircuitTracerBackend(model_name, device=device)
    raise ValueError(f"Unknown backend: {name}")


__all__ = [
    "CircuitTracerBackend",
    "HFHooksBackend",
    "NNsightBackend",
    "TraceBackend",
    "TransformerLensBackend",
    "get_backend",
]
