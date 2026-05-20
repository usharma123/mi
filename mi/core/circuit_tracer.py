from __future__ import annotations

import json
from pathlib import Path

from mi.core.schema import BehaviorSpec, GraphArtifact


def import_circuit_tracer_graph(
    *,
    run_id: str,
    path: Path,
    behavior: BehaviorSpec,
) -> GraphArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_nodes = payload.get("nodes", [])
    raw_edges = payload.get("edges", [])
    nodes = []
    edges = []
    for node in raw_nodes:
        node_id = str(node.get("id"))
        kind = str(node.get("kind", node.get("type", "unknown")))
        normalized_kind = _normalize_node_kind(kind)
        nodes.append(
            {
                "id": node_id,
                "kind": normalized_kind,
                "label": str(node.get("label", node_id)),
                "metadata": {k: v for k, v in node.items() if k not in {"id", "kind", "type", "label"}},
            }
        )
    for edge in raw_edges:
        edges.append(
            {
                "source": str(edge.get("source")),
                "target": str(edge.get("target")),
                "weight": float(edge.get("weight", edge.get("attribution", 0.0))),
                "evidence_ids": [],
                "method": "circuit_tracer_import",
            }
        )
    return GraphArtifact(
        id=run_id,
        backend="circuit-tracer",
        behavior=behavior,
        nodes=nodes,
        edges=edges,
        warnings=["Imported circuit-tracer graph; validate claims with interventions before trusting labels."],
    )


def _normalize_node_kind(kind: str) -> str:
    normalized = kind.lower().replace("-", "_")
    if "transcoder" in normalized:
        return "transcoder_feature"
    if "sae" in normalized or "feature" in normalized:
        return "sae_feature"
    if "error" in normalized:
        return "error"
    if "logit" in normalized:
        return "output_logit"
    if "token" in normalized:
        return "input_token"
    return "residual_stream"
