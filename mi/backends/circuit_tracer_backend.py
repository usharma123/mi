from __future__ import annotations

from mi.backends.base import BackendCapabilityError, NotImplementedTraceBackend


class CircuitTracerBackend(NotImplementedTraceBackend):
    name = "circuit-tracer"

    def trace(self, *args, **kwargs):
        raise BackendCapabilityError(
            "circuit-tracer is installed as a first-class dependency, but graph import "
            "is exposed through `mi graph --backend circuit-tracer --import`."
        )

    def capabilities(self) -> dict[str, bool]:
        return {"trace": False, "localize": False, "features": False, "graph": True}
