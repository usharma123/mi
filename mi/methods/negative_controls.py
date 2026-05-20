from __future__ import annotations


class NegativeControlsNotImplemented(NotImplementedError):
    pass


def require_negative_controls() -> None:
    raise NegativeControlsNotImplemented("Negative controls are planned for mi v0.2.")
