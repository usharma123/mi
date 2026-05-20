from __future__ import annotations


class AttributionPatchingNotImplemented(NotImplementedError):
    pass


def require_attribution_patching() -> None:
    raise AttributionPatchingNotImplemented(
        "Attribution patching is planned for mi v0.2."
    )
