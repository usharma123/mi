from __future__ import annotations

import numpy as np

from mi.backends.transformer_lens_backend import TransformerLensBackend
from mi.core.schema import ActivationRef, LocalizationCandidate


def test_control_summary_sampling_is_deterministic() -> None:
    backend = TransformerLensBackend("fake-model", device="cpu")
    candidates = [
        LocalizationCandidate(
            method="zero_ablation",
            target=ActivationRef(layer=0, position=2, stream="resid_post"),
            hook_name="blocks.0.hook_resid_post",
            metric_before=5.0,
            metric_after=1.0,
            effect=4.0,
        ),
        LocalizationCandidate(
            method="zero_ablation",
            target=ActivationRef(layer=0, position=2, stream="mlp_out"),
            hook_name="blocks.0.hook_mlp_out",
            metric_before=5.0,
            metric_after=4.0,
            effect=1.0,
        ),
        LocalizationCandidate(
            method="zero_ablation",
            target=ActivationRef(layer=1, position=2, stream="resid_post"),
            hook_name="blocks.1.hook_resid_post",
            metric_before=5.0,
            metric_after=3.0,
            effect=2.0,
        ),
    ]

    first = backend._attach_control_summaries(
        candidates,
        controls={"random", "same-layer"},
        control_effects_by_key={},
        rng=np.random.default_rng(0),
    )
    second = backend._attach_control_summaries(
        candidates,
        controls={"random", "same-layer"},
        control_effects_by_key={},
        rng=np.random.default_rng(0),
    )

    assert first[0].control_summary == second[0].control_summary
    assert first[0].control_summary.control_count > 0
