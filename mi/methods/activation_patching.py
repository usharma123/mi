from __future__ import annotations


class ActivationPatchingNotImplemented(NotImplementedError):
    pass


def require_activation_patching() -> None:
    raise ActivationPatchingNotImplemented(
        "Activation patching is planned for mi v0.2. v0.1 captures activations and "
        "reports trace-time attribution summaries only."
    )
