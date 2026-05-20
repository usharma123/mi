from __future__ import annotations

from mi.core.schema import (
    ActivationRef,
    BehaviorSpec,
    Claim,
    ClaimSpec,
    ClaimTestSpec,
    ControlSummary,
    Evidence,
    FeatureRef,
    Intervention,
    LocalizationArtifact,
    LocalizationCandidate,
    ValidationArtifact,
    ValidationResult,
)


def test_behavior_spec_round_trips_json() -> None:
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital of France is",
        target_text=" Paris",
        target_token=" Paris",
    )

    restored = BehaviorSpec.model_validate_json(behavior.model_dump_json())

    assert restored == behavior


def test_claim_and_evidence_schema_round_trip() -> None:
    target = ActivationRef(layer=3, position=5, stream="resid_post")
    evidence = Evidence(
        id="ev_1",
        method="activation_ablation",
        target=target,
        metric_before=10.0,
        metric_after=8.5,
        delta=-1.5,
        controls=["same-layer-random"],
        artifact_refs=["metrics.json"],
    )
    claim = Claim(
        id="claim_1",
        text="L3 residual stream contributes to the target token.",
        target=target,
        evidence_ids=[evidence.id],
        confidence="medium",
        verdict="supported",
    )

    assert Evidence.model_validate_json(evidence.model_dump_json()) == evidence
    assert Claim.model_validate_json(claim.model_dump_json()) == claim


def test_feature_labels_are_metadata_not_evidence() -> None:
    feature = FeatureRef(
        dictionary_id="saelens/gpt2-small",
        layer=7,
        feature_id=123,
        source="sae",
        label="France geography",
        metadata={"neuronpedia_url": "https://example.test"},
    )
    intervention = Intervention(kind="zero", target=feature)

    assert intervention.target.label == "France geography"
    assert intervention.target.metadata["neuronpedia_url"] == "https://example.test"


def test_localization_artifact_round_trips_json() -> None:
    behavior = BehaviorSpec(model="gpt2-small", prompt="Hello", target_text=" world")
    candidate = LocalizationCandidate(
        method="zero_ablation",
        target=ActivationRef(layer=0, position=2, stream="resid_post"),
        hook_name="blocks.0.hook_resid_post",
        metric_before=2.0,
        metric_after=1.0,
        effect=1.0,
        rank=1,
        controls=["random"],
        control_summary=ControlSummary(
            control_mean=0.1,
            control_max=0.2,
            control_count=2,
            specificity_passed=True,
            control_effects={"random": [0.1, 0.2]},
        ),
    )
    artifact = LocalizationArtifact(
        id="run-localize",
        backend="transformer-lens",
        behavior=behavior,
        candidates=[candidate],
    )

    assert LocalizationArtifact.model_validate_json(artifact.model_dump_json()) == artifact


def test_validation_artifact_round_trips_json() -> None:
    behavior = BehaviorSpec(model="gpt2-small", prompt="Hello", target_text=" world")
    claim = ClaimSpec(
        id="claim_1",
        text="A component supports the target token.",
        behavior=behavior,
        target=ActivationRef(layer=0, position=2, stream="resid_post"),
        tests=[ClaimTestSpec(method="zero_ablation", min_effect=0.5)],
    )
    result = ValidationResult(claim_id=claim.id, verdict="untested")
    artifact = ValidationArtifact(
        id="validation",
        backend="transformer-lens",
        claims=[claim],
        results=[result],
    )

    assert ValidationArtifact.model_validate_json(artifact.model_dump_json()) == artifact
