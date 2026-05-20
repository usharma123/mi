from __future__ import annotations


class CircuitGraphNotImplemented(NotImplementedError):
    pass


def require_circuit_graph() -> None:
    raise CircuitGraphNotImplemented("Evidence graph construction is planned for mi v0.4.")
