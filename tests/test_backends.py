from __future__ import annotations

import pytest

from mi.backends import get_backend
from mi.backends.base import BackendCapabilityError
from mi.core.schema import BehaviorSpec


def test_unsupported_backends_raise_capability_errors(tmp_path) -> None:
    for name in ("nnsight", "hf-hooks", "circuit-tracer"):
        backend = get_backend(name)("fake", None)
        with pytest.raises(BackendCapabilityError):
            backend.trace(
                BehaviorSpec(model="fake", prompt="hello"),
                tmp_path / "activations.npz",
                run_id="run",
                top_k=1,
            )
