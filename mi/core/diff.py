from __future__ import annotations

from mi.core.schema import DiffArtifact, TraceArtifact


def diff_traces(run_id: str, trace_a: TraceArtifact, trace_b: TraceArtifact) -> DiffArtifact:
    output_deltas = []
    top_a = trace_a.top_predictions[0] if trace_a.top_predictions else None
    top_b = trace_b.top_predictions[0] if trace_b.top_predictions else None
    output_deltas.append(
        {
            "prompt": trace_a.behavior.prompt,
            "model_a_top": top_a.token if top_a else None,
            "model_b_top": top_b.token if top_b else None,
            "target_logit_delta": (
                trace_b.target.logit - trace_a.target.logit
                if trace_a.target is not None and trace_b.target is not None
                else None
            ),
            "target_rank_delta": (
                trace_b.target.rank - trace_a.target.rank
                if trace_a.target is not None and trace_b.target is not None
                else None
            ),
        }
    )
    return DiffArtifact(
        id=run_id,
        model_a=trace_a.behavior.model,
        model_b=trace_b.behavior.model,
        prompt_count=1,
        output_deltas=output_deltas,
    )
