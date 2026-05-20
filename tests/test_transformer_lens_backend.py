from __future__ import annotations

import importlib.util

import pytest

from mi.backends.transformer_lens_backend import TransformerLensBackend
from mi.core.schema import BehaviorSpec


@pytest.mark.integration
def test_transformer_lens_backend_smoke(tmp_path) -> None:
    if importlib.util.find_spec("transformer_lens") is None:
        pytest.skip("transformer_lens is not installed")

    backend = TransformerLensBackend("gpt2-small", device="cpu")
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital of France is",
        target_text=" Paris",
    )

    trace = backend.trace(
        behavior,
        tmp_path / "activations.npz",
        run_id="integration-run",
        top_k=5,
    )

    assert trace.top_predictions
    assert trace.target is not None
    assert (tmp_path / "activations.npz").exists()
