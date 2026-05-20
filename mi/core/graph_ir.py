from __future__ import annotations

from typing import Literal

from pydantic import Field

from mi.core.schema import MIModel


class GraphNode(MIModel):
    id: str
    kind: Literal[
        "input_token",
        "attention_head",
        "residual_stream",
        "sae_feature",
        "transcoder_feature",
        "mlp_output",
        "error",
        "output_logit",
    ]
    label: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class GraphEdge(MIModel):
    source: str
    target: str
    weight: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceGraph(MIModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
