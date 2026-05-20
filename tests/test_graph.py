from __future__ import annotations

from mi.core.graph_builder import build_meir_graph, graph_to_graphml
from mi.core.schema import (
    BehaviorSpec,
    DirectLogitAttributionEntry,
    FeatureArtifact,
    FeatureCandidate,
    FeatureRef,
    TraceArtifact,
)
from mi.report import render_graph_markdown


def test_build_meir_graph_from_trace_and_features() -> None:
    behavior = BehaviorSpec(model="fake", prompt="hello", target_text=" world", target_token=" world")
    trace = TraceArtifact(
        id="trace",
        backend="transformer-lens",
        behavior=behavior,
        token_ids=[1],
        tokens=["hello"],
        direct_logit_attribution=[
            DirectLogitAttributionEntry(layer=0, component="mlp_out", position=0, contribution=1.0)
        ],
    )
    features = FeatureArtifact(
        id="features",
        backend="transformer-lens",
        behavior=behavior,
        dictionary_id="raw/fake",
        features=[
            FeatureCandidate(
                feature=FeatureRef(
                    dictionary_id="raw/fake",
                    layer=0,
                    feature_id=3,
                    source="custom_direction",
                ),
                activation_value=2.0,
            )
        ],
    )

    graph = build_meir_graph(run_id="graph", trace=trace, features=features)

    assert graph.nodes
    assert graph.edges
    assert "graphml" in graph_to_graphml(graph)
    assert "Evidence Graph Report" in render_graph_markdown(graph)
