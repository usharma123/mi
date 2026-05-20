from __future__ import annotations

from mi.backends.base import NotImplementedTraceBackend


class NNsightBackend(NotImplementedTraceBackend):
    name = "nnsight"

    def trace(self, *args, **kwargs):
        raise NotImplementedError(
            "NNsight is installed as a first-class dependency, but the NNsight backend "
            "is planned for mi v0.7. Use --backend transformer-lens in v0.1."
        )
