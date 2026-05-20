from __future__ import annotations

from mi.core.cts import score_validation
from mi.core.fuzz import generate_variants
from mi.core.schema import ActivationRef, ClaimTestResult, ValidationArtifact, ValidationResult


def test_generate_variants_from_family() -> None:
    variants = generate_variants(
        {
            "family": "capital_cities",
            "templates": ["The capital of {country} is"],
            "variables": {"country": ["France", "Germany"]},
            "target_template": " Paris",
        }
    )

    assert [variant.prompt for variant in variants] == [
        "The capital of France is",
        "The capital of Germany is",
    ]
    assert variants[0].target_text == " Paris"


def test_score_validation_computes_cts() -> None:
    validation = ValidationArtifact(
        id="validation",
        backend="transformer-lens",
        results=[
            ValidationResult(
                claim_id="claim",
                verdict="supported",
                tests=[
                    ClaimTestResult(
                        method="zero_ablation",
                        target=ActivationRef(layer=0, position=0, stream="resid_post"),
                        passed=True,
                        effect=5.0,
                        min_effect=1.0,
                        reason="ok",
                    )
                ],
            )
        ],
    )

    scores = score_validation(validation)

    assert scores.scores["claim"]["cts"] is not None
