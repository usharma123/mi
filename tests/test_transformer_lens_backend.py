from __future__ import annotations

import importlib.util

import pytest

from mi.backends.transformer_lens_backend import TransformerLensBackend
from mi.core.schema import BehaviorSpec, ClaimSpec, ClaimTestSpec
from mi.core.validation import evaluate_claim
from mi.methods.features import build_saelens_features


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


@pytest.mark.integration
def test_transformer_lens_localize_smoke() -> None:
    if importlib.util.find_spec("transformer_lens") is None:
        pytest.skip("transformer_lens is not installed")

    backend = TransformerLensBackend("gpt2-small", device="cpu")
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital of France is",
        target_text=" Paris",
    )

    localization = backend.localize(
        behavior,
        run_id="integration-localize",
        corrupt_prompt="The capital of Germany is",
        methods={"zero_ablation", "clean_to_corrupt_patch"},
        streams={"resid_post"},
        controls={"random", "same-layer"},
        position="final",
        seed=0,
        top_k=2,
    )

    assert localization.candidates
    assert any(item.method == "zero_ablation" for item in localization.candidates)
    assert any(item.method == "clean_to_corrupt_patch" for item in localization.candidates)
    assert any(item.control_summary is not None for item in localization.candidates)


@pytest.mark.integration
def test_transformer_lens_validation_smoke() -> None:
    if importlib.util.find_spec("transformer_lens") is None:
        pytest.skip("transformer_lens is not installed")

    backend = TransformerLensBackend("gpt2-small", device="cpu")
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital city of France is called",
        target_text=" Paris",
    )
    localization = backend.localize(
        behavior,
        run_id="integration-validate",
        corrupt_prompt="The capital city of Germany is called",
        methods={"clean_to_corrupt_patch"},
        streams={"resid_post"},
        controls={"random"},
        position="final",
        seed=0,
        top_k=1,
    )
    candidate = localization.candidates[0]
    claim = ClaimSpec(
        id="integration_claim",
        text="Top localized candidate has a positive patch effect.",
        target=candidate.target,
        hook_name=candidate.hook_name,
        tests=[ClaimTestSpec(method=candidate.method, min_effect=0.1)],
    )

    result, evidence = evaluate_claim(
        claim,
        [(claim.tests[0], candidate)],
        evidence_start=1,
    )

    assert result.verdict == "supported"
    assert evidence


@pytest.mark.integration
def test_saelens_feature_extraction_smoke(tmp_path) -> None:
    if importlib.util.find_spec("transformer_lens") is None or importlib.util.find_spec("sae_lens") is None:
        pytest.skip("transformer_lens and sae_lens are required")

    backend = TransformerLensBackend("gpt2-small", device="cpu")
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital city of France is called",
        target_text=" Paris",
    )
    activation_path = tmp_path / "activations.npz"
    backend.trace(behavior, activation_path, run_id="integration-sae-trace", top_k=3)

    features = build_saelens_features(
        run_id="integration-sae-features",
        backend="transformer-lens",
        behavior=behavior,
        activation_path=activation_path,
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        hook_name="blocks.8.hook_resid_pre",
        top_k=1,
        device="cpu",
    )

    assert features.features
    assert features.sae_feature_dim is not None
    assert features.features[0].dictionary_reconstruction_error is not None
