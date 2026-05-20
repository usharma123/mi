from __future__ import annotations

from mi.core.diff import diff_traces
from mi.core.schema import BehaviorSpec, TargetMetrics, TopPrediction, TraceArtifact
from mi.report import render_diff_markdown


def test_diff_traces_reports_output_delta() -> None:
    behavior_a = BehaviorSpec(model="a", prompt="hello", target_text=" world")
    behavior_b = BehaviorSpec(model="b", prompt="hello", target_text=" world")
    trace_a = TraceArtifact(
        id="a",
        backend="transformer-lens",
        behavior=behavior_a,
        token_ids=[1],
        tokens=["hello"],
        target=TargetMetrics(token_id=2, token=" world", logit=1.0, probability=0.1, rank=3),
        top_predictions=[TopPrediction(token_id=2, token=" world", logit=1.0, probability=0.1, rank=1)],
    )
    trace_b = trace_a.model_copy(
        update={
            "id": "b",
            "behavior": behavior_b,
            "target": TargetMetrics(token_id=2, token=" world", logit=2.0, probability=0.2, rank=1),
        }
    )

    diff = diff_traces("diff", trace_a, trace_b)

    assert diff.output_deltas[0]["target_logit_delta"] == 1.0
    assert "Model Diff Report" in render_diff_markdown(diff)
