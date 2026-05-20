from __future__ import annotations

from typing import Any

from mi.core.schema import TraceArtifact


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
