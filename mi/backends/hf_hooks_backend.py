from __future__ import annotations

from mi.backends.base import BackendCapabilityError, NotImplementedTraceBackend


class HFHooksBackend(NotImplementedTraceBackend):
    name = "hf-hooks"

    def trace(self, *args, **kwargs):
        raise BackendCapabilityError(
            "The raw Hugging Face hooks backend requires model-specific hook maps before "
            "causal trace/localize can run. Use --backend transformer-lens for supported models."
        )

    def capabilities(self) -> dict[str, bool]:
        return {"trace": False, "localize": False, "features": False, "graph": False}
