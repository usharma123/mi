from __future__ import annotations

from typing import Any

from mi.core.schema import LocalizationArtifact, TraceArtifact


def render_json_report(trace: TraceArtifact) -> dict[str, Any]:
    return {
        "run_id": trace.id,
        "behavior": trace.behavior.model_dump(mode="json"),
        "backend": trace.backend,
        "target": trace.target.model_dump(mode="json") if trace.target else None,
        "top_predictions": [item.model_dump(mode="json") for item in trace.top_predictions],
        "logit_lens": [item.model_dump(mode="json") for item in trace.logit_lens],
        "direct_logit_attribution": [
            item.model_dump(mode="json") for item in trace.direct_logit_attribution
        ],
        "activation_inventory": [
            item.model_dump(mode="json") for item in trace.activation_inventory
        ],
        "warnings": trace.warnings,
        "artifact_refs": trace.artifact_refs,
    }


def render_localization_json_report(localization: LocalizationArtifact) -> dict[str, Any]:
    return {
        "run_id": localization.id,
        "behavior": localization.behavior.model_dump(mode="json"),
        "backend": localization.backend,
        "corrupt_prompt": localization.corrupt_prompt,
        "target": localization.target.model_dump(mode="json") if localization.target else None,
        "corrupt_target": localization.corrupt_target.model_dump(mode="json")
        if localization.corrupt_target
        else None,
        "candidates": [item.model_dump(mode="json") for item in localization.candidates],
        "evidence": [item.model_dump(mode="json") for item in localization.evidence],
        "warnings": localization.warnings,
        "artifact_refs": localization.artifact_refs,
    }
