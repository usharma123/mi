from __future__ import annotations

from mi.core.regression import claim_paths_from_globs, regression_exit_code
from mi.core.schema import ValidationArtifact, ValidationResult


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
