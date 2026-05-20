from __future__ import annotations

import pytest

from mi.core.schema import (
    ActivationRef,
    ActivationSummary,
    BehaviorSpec,
    DirectLogitAttributionEntry,
    LogitLensEntry,
    TargetMetrics,
    TopPrediction,
    TraceArtifact,
)


@pytest.fixture
def sample_trace():
    return make_sample_trace


def make_sample_trace(run_id: str = "sample-run") -> TraceArtifact:
    behavior = BehaviorSpec(
        model="gpt2-small",
        prompt="The capital of France is",
        target_text=" Paris",
        target_token=" Paris",
    )
    return TraceArtifact(
        id=run_id,
        backend="transformer-lens",
        behavior=behavior,
        token_ids=[464, 3139, 286, 4881, 318],
        tokens=["The", " capital", " of", " France", " is"],
        target=TargetMetrics(token_id=6342, token=" Paris", logit=18.4, probability=0.71, rank=1),
        top_predictions=[
            TopPrediction(token_id=6342, token=" Paris", logit=18.4, probability=0.71, rank=1),
            TopPrediction(token_id=198, token="\n", logit=12.2, probability=0.02, rank=2),
        ],
        logit_lens=[
            LogitLensEntry(
                layer=0,
                stream="resid_post",
                target_logit=1.2,
                target_rank=100,
                top_token=" the",
                top_token_id=262,
                top_logit=4.2,
            )
        ],
        direct_logit_attribution=[
            DirectLogitAttributionEntry(layer=0, component="mlp_out", position=4, contribution=0.5)
        ],
        activation_inventory=[
            ActivationSummary(
                name="blocks.0.hook_resid_post",
                shape=[1, 5, 768],
                dtype="float32",
                artifact_key="blocks__0__hook_resid_post",
                ref=ActivationRef(layer=0, position=4, stream="resid_post"),
            )
        ],
    )
