from __future__ import annotations

from mi.core.regression import claim_paths_from_globs, regression_exit_code, validation_for_claims
from mi.core.schema import ActivationRef, ClaimSpec, ValidationArtifact, ValidationResult


def test_regression_exit_code_uses_worst_verdict() -> None:
    validation = ValidationArtifact(
        id="test",
        backend="offline",
        results=[
            ValidationResult(claim_id="a", verdict="supported"),
            ValidationResult(claim_id="b", verdict="contradicted"),
        ],
    )

    assert regression_exit_code(validation) == 4


def test_claim_paths_from_globs(tmp_path, monkeypatch) -> None:
    claim = tmp_path / "claim.yml"
    claim.write_text("id: test\ntext: t\ntarget: {layer: 0, position: 0, stream: resid_post}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert claim_paths_from_globs(("*.yml",)) == [claim.relative_to(tmp_path)]


def test_validation_for_claims_marks_missing_claims_untested() -> None:
    validation = ValidationArtifact(
        id="test",
        backend="transformer-lens",
        results=[ValidationResult(claim_id="covered", verdict="supported")],
    )
    claims = [
        ClaimSpec(
            id="covered",
            text="Covered claim.",
            target=ActivationRef(layer=0, position=0, stream="resid_post"),
        ),
        ClaimSpec(
            id="missing",
            text="Missing claim.",
            target=ActivationRef(layer=1, position=0, stream="resid_post"),
        ),
    ]

    scoped = validation_for_claims(validation, claims)

    assert [result.verdict for result in scoped.results] == ["supported", "untested"]
    assert regression_exit_code(scoped) == 3
    assert "missing" in scoped.warnings[-1]
