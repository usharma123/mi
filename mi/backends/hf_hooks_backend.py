from __future__ import annotations

from mi.backends.base import NotImplementedTraceBackend


class HFHooksBackend(NotImplementedTraceBackend):
    name = "hf-hooks"

    def trace(self, *args, **kwargs):
        raise NotImplementedError(
            "The raw Hugging Face hooks backend is planned for a later release. "
            "Use --backend transformer-lens in mi v0.1."
        )
