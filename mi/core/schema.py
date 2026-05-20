from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BehaviorSpec(MIModel):
    model: str
    prompt: str
    target_token: str | None = None
    target_text: str | None = None
    clean_prompt: str | None = None
    corrupt_prompt: str | None = None
    metric: Literal["target_logit", "logit_diff", "prob", "rank", "kl"] = "target_logit"


class ActivationRef(MIModel):
    layer: int
    position: int
    stream: Literal["resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"]
    component: str | None = None


class FeatureRef(MIModel):
    dictionary_id: str
    layer: int
    feature_id: int
    source: Literal["sae", "transcoder", "raw_neuron", "nla", "custom_direction"]
    label: str | None = Field(
        default=None,
        description="Human-facing metadata only. Labels are not mechanistic evidence.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Non-evidence feature metadata such as examples or registry URLs.",
    )


class Intervention(MIModel):
    kind: Literal["zero", "mean_ablate", "patch", "scale", "set"]
    target: ActivationRef | FeatureRef
    value_source: str | None = None


class Evidence(MIModel):
    id: str
    method: str
    target: ActivationRef | FeatureRef
    metric_before: float
    metric_after: float
    delta: float
    controls: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)


class Claim(MIModel):
    id: str
    text: str
    target: ActivationRef | FeatureRef
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    verdict: Literal["supported", "weak", "contradicted", "untested"] = "untested"


class ControlSummary(MIModel):
    control_mean: float | None = None
    control_max: float | None = None
    control_count: int = 0
    specificity_passed: bool | None = None
    control_effects: dict[str, list[float]] = Field(default_factory=dict)


class TopPrediction(MIModel):
    token_id: int
    token: str
    logit: float
    probability: float
    rank: int


class TargetMetrics(MIModel):
    token_id: int
    token: str
    logit: float
    probability: float
    rank: int


class LogitLensEntry(MIModel):
    layer: int
    stream: Literal["resid_pre", "resid_mid", "resid_post", "final"]
    target_logit: float | None = None
    target_rank: int | None = None
    top_token: str | None = None
    top_token_id: int | None = None
    top_logit: float | None = None


class DirectLogitAttributionEntry(MIModel):
    layer: int | None = None
    component: str
    position: int
    contribution: float


class LocalizationCandidate(MIModel):
    method: Literal["zero_ablation", "clean_to_corrupt_patch"]
    target: ActivationRef
    hook_name: str
    metric_before: float
    metric_after: float
    effect: float
    rank: int | None = None
    controls: list[str] = Field(default_factory=list)
    control_summary: ControlSummary | None = None


class ActivationSummary(MIModel):
    name: str
    shape: list[int]
    dtype: str
    artifact_key: str
    ref: ActivationRef | None = None


class TraceArtifact(MIModel):
    id: str
    created_at: str = Field(default_factory=utc_now_iso)
    backend: str
    behavior: BehaviorSpec
    token_ids: list[int]
    tokens: list[str]
    target: TargetMetrics | None = None
    top_predictions: list[TopPrediction] = Field(default_factory=list)
    logit_lens: list[LogitLensEntry] = Field(default_factory=list)
    direct_logit_attribution: list[DirectLogitAttributionEntry] = Field(default_factory=list)
    activation_inventory: list[ActivationSummary] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class LocalizationArtifact(MIModel):
    id: str
    created_at: str = Field(default_factory=utc_now_iso)
    backend: str
    behavior: BehaviorSpec
    corrupt_prompt: str | None = None
    target: TargetMetrics | None = None
    corrupt_target: TargetMetrics | None = None
    candidates: list[LocalizationCandidate] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ClaimTestSpec(MIModel):
    method: Literal["zero_ablation", "clean_to_corrupt_patch"]
    min_effect: float = Field(default=0.0, ge=0)
    max_control_effect: float | None = Field(default=None, ge=0)


class ClaimSpec(MIModel):
    id: str
    text: str
    target: ActivationRef
    behavior: BehaviorSpec | None = None
    hook_name: str | None = None
    corrupt_prompt: str | None = None
    tests: list[ClaimTestSpec] = Field(default_factory=list)


class ClaimTestResult(MIModel):
    method: str
    target: ActivationRef
    passed: bool
    effect: float | None = None
    min_effect: float | None = None
    control_max: float | None = None
    max_control_effect: float | None = None
    evidence_id: str | None = None
    reason: str


class ValidationResult(MIModel):
    claim_id: str
    verdict: Literal["supported", "weak", "contradicted", "untested"]
    tests: list[ClaimTestResult] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ValidationArtifact(MIModel):
    id: str
    created_at: str = Field(default_factory=utc_now_iso)
    backend: str
    claims: list[ClaimSpec] = Field(default_factory=list)
    results: list[ValidationResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    artifact_refs: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RunManifest(MIModel):
    run_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    version: str
    backend: str
    behavior: BehaviorSpec
    artifacts: dict[str, str] = Field(default_factory=dict)
