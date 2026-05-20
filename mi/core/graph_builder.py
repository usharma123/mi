from __future__ import annotations

from mi.core.schema import (
    FeatureArtifact,
    GraphArtifact,
    LocalizationArtifact,
    TraceArtifact,
)


def build_meir_graph(
    *,
    run_id: str,
    trace: TraceArtifact,
    localization: LocalizationArtifact | None = None,
    features: FeatureArtifact | None = None,
    prune_threshold: float = 0.0,
) -> GraphArtifact:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    evidence = []

    def add_node(node_id: str, kind: str, label: str, **metadata) -> None:
        nodes.setdefault(
            node_id,
            {"id": node_id, "kind": kind, "label": label, "metadata": metadata},
        )

    for position, token in enumerate(trace.tokens):
        add_node(f"token:{position}", "input_token", token, position=position)
    add_node("logit:target", "output_logit", trace.behavior.target_token or trace.behavior.target_text or "target")

    for item in trace.direct_logit_attribution:
        if abs(item.contribution) < prune_threshold:
            continue
        node_id = f"component:L{item.layer}:{item.component}:P{item.position}"
        add_node(node_id, "mlp_output" if item.component == "mlp_out" else "attention_head", node_id)
        edges.append(
            {
                "source": node_id,
                "target": "logit:target",
                "weight": item.contribution,
                "evidence_ids": [],
                "method": "direct_logit_attribution",
            }
        )

    if localization is not None:
        evidence.extend(localization.evidence)
        for candidate in localization.candidates:
            if abs(candidate.effect) < prune_threshold:
                continue
            node_id = (
                f"activation:L{candidate.target.layer}:"
                f"{candidate.target.stream}:P{candidate.target.position}"
            )
            add_node(node_id, _stream_kind(candidate.target.stream), node_id)
            edges.append(
                {
                    "source": node_id,
                    "target": "logit:target",
                    "weight": candidate.effect,
                    "evidence_ids": [],
                    "method": candidate.method,
                }
            )

    if features is not None:
        evidence.extend(features.evidence)
        for feature in features.features:
            if abs(feature.activation_value) < prune_threshold:
                continue
            node_id = (
                f"feature:{feature.feature.source}:L{feature.feature.layer}:"
                f"{feature.feature.feature_id}"
            )
            add_node(
                node_id,
                "sae_feature" if feature.feature.source == "sae" else "residual_stream",
                feature.feature.label or node_id,
                dictionary_id=feature.feature.dictionary_id,
                metadata_only=True,
            )
            edges.append(
                {
                    "source": node_id,
                    "target": "logit:target",
                    "weight": feature.activation_value,
                    "evidence_ids": feature.evidence_ids,
                    "method": "feature_activation",
                }
            )

    return GraphArtifact(
        id=run_id,
        backend=trace.backend,
        behavior=trace.behavior,
        nodes=list(nodes.values()),
        edges=edges,
        evidence=evidence,
    )


def graph_to_graphml(graph: GraphArtifact) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <graph edgedefault="directed">',
    ]
    for node in graph.nodes:
        lines.append(f'    <node id="{_xml_escape(node["id"])}" />')
    for index, edge in enumerate(graph.edges):
        lines.append(
            f'    <edge id="e{index}" source="{_xml_escape(edge["source"])}" '
            f'target="{_xml_escape(edge["target"])}" />'
        )
    lines.extend(["  </graph>", "</graphml>"])
    return "\n".join(lines) + "\n"


def _stream_kind(stream: str) -> str:
    if stream == "attn_out":
        return "attention_head"
    if stream == "mlp_out":
        return "mlp_output"
    return "residual_stream"


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
