from __future__ import annotations

from mi.backends.base import BackendCapabilityError, NotImplementedTraceBackend


class NNsightBackend(NotImplementedTraceBackend):
    name = "nnsight"

    def trace(self, *args, **kwargs):
        raise BackendCapabilityError(
            "NNsight is installed as a first-class dependency, but the NNsight backend "
            "does not yet expose stable mi trace/localize semantics. Use --backend transformer-lens."
        )

    def capabilities(self) -> dict[str, bool]:
        return {"trace": False, "localize": False, "features": False, "graph": False}
