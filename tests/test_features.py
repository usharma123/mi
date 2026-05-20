from __future__ import annotations

import numpy as np

from mi.core.schema import BehaviorSpec
from mi.methods.features import build_raw_activation_features
from mi.report import render_features_markdown


def test_build_raw_activation_features_from_npz(tmp_path) -> None:
    activation_path = tmp_path / "activations.npz"
    np.savez_compressed(
        activation_path,
        blocks__0__hook_resid_post=np.array([[[0.1, -2.0, 0.3]]]),
        blocks__1__hook_mlp_out=np.array([[[4.0, 0.2, -0.1]]]),
    )
    behavior = BehaviorSpec(model="fake", prompt="hello", target_text=" world")

    artifact = build_raw_activation_features(
        run_id="run-features",
        backend="transformer-lens",
        behavior=behavior,
        activation_path=activation_path,
        dictionary_id="raw-activation/fake",
        source="raw_activation",
        top_k=2,
    )

    assert len(artifact.features) == 2
    assert artifact.features[0].activation_value == 4.0
    assert artifact.features[0].feature.metadata["metadata_only"] is True
    assert artifact.evidence
    assert "Feature Report" in render_features_markdown(artifact)
