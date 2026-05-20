from __future__ import annotations

import json

from mi.core.circuit_tracer import import_circuit_tracer_graph
from mi.core.schema import BehaviorSpec


def test_import_circuit_tracer_graph(tmp_path) -> None:
    path = tmp_path / "ct.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "f1", "kind": "transcoder_feature", "label": "feature"}],
                "edges": [{"source": "f1", "target": "logit", "attribution": 0.5}],
            }
        ),
        encoding="utf-8",
    )

    graph = import_circuit_tracer_graph(
        run_id="ct",
        path=path,
        behavior=BehaviorSpec(model="fake", prompt="hello"),
    )

    assert graph.backend == "circuit-tracer"
    assert graph.nodes[0]["kind"] == "transcoder_feature"
    assert graph.edges[0]["weight"] == 0.5
