from __future__ import annotations

import json
from pathlib import Path

import yaml

from mi.core.schema import (
    ClaimSpec,
    ClaimTestResult,
    ClaimTestSpec,
    Evidence,
    LocalizationArtifact,
    LocalizationCandidate,
    ValidationResult,
)


def load_claim_specs(path: Path) -> list[ClaimSpec]:
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        payload = json.loads(raw_text)
    elif suffix in {".yml", ".yaml"}:
        payload = yaml.safe_load(raw_text)
    else:
        raise ValueError("Claim files must be .json, .yml, or .yaml")

    if isinstance(payload, dict) and "claims" in payload:
        items = payload["claims"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValueError("Claim file must contain a claim object, a list, or {claims: [...]}.")
    return [ClaimSpec.model_validate(item) for item in items]


def find_candidate(
    localization: LocalizationArtifact,
    claim: ClaimSpec,
    test: ClaimTestSpec,
) -> LocalizationCandidate | None:
    for candidate in localization.candidates:
        if candidate.method != test.method:
            continue
        if candidate.target != claim.target:
            continue
        if claim.hook_name and candidate.hook_name != claim.hook_name:
            continue
        return candidate
    return None


def evaluate_claim(
    claim: ClaimSpec,
    tests: list[tuple[ClaimTestSpec, LocalizationCandidate | None]],
    *,
    evidence_start: int,
) -> tuple[ValidationResult, list[Evidence]]:
    results: list[ClaimTestResult] = []
    evidence: list[Evidence] = []
    for index, (test, candidate) in enumerate(tests, start=evidence_start):
        if candidate is None:
            results.append(
                ClaimTestResult(
                    method=test.method,
                    target=claim.target,
                    passed=False,
                    min_effect=test.min_effect,
                    max_control_effect=test.max_control_effect,
                    reason="No matching localization candidate was found.",
                )
            )
            continue

        control_max = (
            candidate.control_summary.control_max
            if candidate.control_summary is not None
            else None
        )
        effect_passed = candidate.effect >= test.min_effect
        control_passed = (
            True
            if test.max_control_effect is None
            else control_max is not None and control_max <= test.max_control_effect
        )
        passed = effect_passed and control_passed
        if not effect_passed:
            reason = f"Effect {candidate.effect:.4f} is below minimum {test.min_effect:.4f}."
        elif not control_passed:
            reason = (
                "Control max is missing or above threshold "
                f"{test.max_control_effect:.4f}."
            )
        else:
            reason = "Test passed."

        evidence_id = f"val_ev_{index}"
        evidence.append(
            Evidence(
                id=evidence_id,
                method=test.method,
                target=candidate.target,
                metric_before=candidate.metric_before,
                metric_after=candidate.metric_after,
                delta=candidate.effect,
                controls=candidate.controls,
                artifact_refs=["validation.json"],
            )
        )
        results.append(
            ClaimTestResult(
                method=test.method,
                target=candidate.target,
                passed=passed,
                effect=candidate.effect,
                min_effect=test.min_effect,
                control_max=control_max,
                max_control_effect=test.max_control_effect,
                evidence_id=evidence_id,
                reason=reason,
            )
        )

    verdict = verdict_for_results(results)
    return (
        ValidationResult(
            claim_id=claim.id,
            verdict=verdict,
            tests=results,
            evidence_ids=[item.id for item in evidence],
        ),
        evidence,
    )


def verdict_for_results(results: list[ClaimTestResult]) -> str:
    if not results or all(result.effect is None for result in results):
        return "untested"
    if all(result.passed for result in results):
        return "supported"
    if any(result.effect is not None and result.effect <= 0 for result in results):
        return "contradicted"
    return "weak"
