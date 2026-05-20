from __future__ import annotations

from mi.backends.base import NotImplementedTraceBackend


class CircuitTracerBackend(NotImplementedTraceBackend):
    name = "circuit-tracer"

    def trace(self, *args, **kwargs):
        raise NotImplementedError(
            "circuit-tracer is installed as a first-class dependency, but graph import "
            "and validation are planned for mi v0.6. Use --backend transformer-lens in v0.1."
        )
