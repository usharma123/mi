from __future__ import annotations

from mi.core.schema import (
    ActivationRef,
    ClaimSpec,
    ClaimTestSpec,
    ControlSummary,
    LocalizationArtifact,
    LocalizationCandidate,
)
from mi.core.validation import apply_variant_threshold, evaluate_claim, find_candidate, find_variant_candidate


def test_find_candidate_matches_method_target_and_hook() -> None:
    target = ActivationRef(layer=0, position=2, stream="resid_post")
    candidate = LocalizationCandidate(
        method="zero_ablation",
        target=target,
        hook_name="blocks.0.hook_resid_post",
        metric_before=2.0,
        metric_after=1.0,
        effect=1.0,
    )
    localization = LocalizationArtifact(
        id="loc",
        backend="transformer-lens",
        behavior={"model": "fake", "prompt": "hello", "target_text": " world"},
        candidates=[candidate],
    )
    claim = ClaimSpec(
        id="claim",
        text="test",
        target=target,
        hook_name="blocks.0.hook_resid_post",
        tests=[ClaimTestSpec(method="zero_ablation", min_effect=0.5)],
    )

    assert find_candidate(localization, claim, claim.tests[0]) == candidate


def test_evaluate_claim_verdicts() -> None:
    target = ActivationRef(layer=0, position=2, stream="resid_post")
    claim = ClaimSpec(
        id="claim",
        text="test",
        target=target,
        tests=[ClaimTestSpec(method="zero_ablation", min_effect=0.5, max_control_effect=0.2)],
    )
    supported_candidate = LocalizationCandidate(
        method="zero_ablation",
        target=target,
        hook_name="blocks.0.hook_resid_post",
        metric_before=2.0,
        metric_after=1.0,
        effect=1.0,
        control_summary=ControlSummary(control_max=0.1, control_count=1),
    )
    weak_candidate = supported_candidate.model_copy(
        update={"effect": 0.25, "metric_after": 1.75}
    )
    contradicted_candidate = supported_candidate.model_copy(
        update={"effect": -0.1, "metric_after": 2.1}
    )

    supported, _ = evaluate_claim(claim, [(claim.tests[0], supported_candidate)], evidence_start=1)
    weak, _ = evaluate_claim(claim, [(claim.tests[0], weak_candidate)], evidence_start=1)
    contradicted, _ = evaluate_claim(
        claim, [(claim.tests[0], contradicted_candidate)], evidence_start=1
    )
    untested, _ = evaluate_claim(claim, [(claim.tests[0], None)], evidence_start=1)

    assert supported.verdict == "supported"
    assert weak.verdict == "weak"
    assert contradicted.verdict == "contradicted"
    assert untested.verdict == "untested"


def test_variant_candidate_matches_layer_stream_without_position() -> None:
    claim = ClaimSpec(
        id="claim",
        text="test",
        target=ActivationRef(layer=0, position=9, stream="resid_post"),
        hook_name="blocks.0.hook_resid_post",
        tests=[ClaimTestSpec(method="zero_ablation")],
    )
    candidate = LocalizationCandidate(
        method="zero_ablation",
        target=ActivationRef(layer=0, position=2, stream="resid_post"),
        hook_name="blocks.0.hook_resid_post",
        metric_before=2.0,
        metric_after=1.0,
        effect=1.0,
    )
    localization = LocalizationArtifact(
        id="loc",
        backend="transformer-lens",
        behavior={"model": "fake", "prompt": "hello", "target_text": " world"},
        candidates=[candidate],
    )

    assert find_variant_candidate(localization, claim, claim.tests[0]) == candidate


def test_apply_variant_threshold_downgrades_supported_claim() -> None:
    claim = ClaimSpec(
        id="claim",
        text="test",
        target=ActivationRef(layer=0, position=0, stream="resid_post"),
        min_variant_pass_rate=0.75,
    )
    result, _ = evaluate_claim(
        claim,
        [
            (
                ClaimTestSpec(method="zero_ablation", min_effect=0.5),
                LocalizationCandidate(
                    method="zero_ablation",
                    target=claim.target,
                    hook_name="blocks.0.hook_resid_post",
                    metric_before=2.0,
                    metric_after=1.0,
                    effect=1.0,
                ),
            )
        ],
        evidence_start=1,
    )

    updated = apply_variant_threshold(
        result,
        claim=claim,
        variant_passed=1,
        variant_total=2,
        variant_contradicted=False,
    )

    assert updated.verdict == "weak"
    assert updated.variant_pass_rate == 0.5
