from __future__ import annotations

from pathlib import Path
from glob import glob

from mi.core.schema import ClaimSpec, ValidationArtifact, ValidationResult


EXIT_CODE_BY_VERDICT = {
    "supported": 0,
    "weak": 2,
    "untested": 3,
    "contradicted": 4,
}


def regression_exit_code(validation: ValidationArtifact) -> int:
    if not validation.results:
        return EXIT_CODE_BY_VERDICT["untested"]
    return max(EXIT_CODE_BY_VERDICT[result.verdict] for result in validation.results)


def validation_for_claims(validation: ValidationArtifact, claims: list[ClaimSpec]) -> ValidationArtifact:
    """Return validation results scoped to the supplied claim specs.

    Missing claims become explicit untested results so CI cannot accidentally pass when a
    validation artifact does not cover a requested claim.
    """
    by_id = {result.claim_id: result for result in validation.results}
    scoped_results: list[ValidationResult] = []
    missing: list[str] = []
    for claim in claims:
        result = by_id.get(claim.id)
        if result is None:
            missing.append(claim.id)
            result = ValidationResult(
                claim_id=claim.id,
                verdict="untested",
                tests=[],
                evidence_ids=[],
            )
        scoped_results.append(result)

    warnings = list(validation.warnings)
    if missing:
        warnings.append(
            "Validation artifact did not include requested claim(s): " + ", ".join(missing)
        )

    return validation.model_copy(update={"claims": claims, "results": scoped_results, "warnings": warnings})


def claim_paths_from_globs(patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate)
            continue
        matches = sorted(Path(match) for match in glob(pattern))
        paths.extend(path for path in matches if path.is_file())
    return paths
