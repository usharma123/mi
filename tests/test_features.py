from __future__ import annotations

import numpy as np
import sys
import types

from mi.core.schema import BehaviorSpec
from mi.methods.features import (
    build_raw_activation_features,
    build_saelens_features,
    resolve_saelens_config,
)
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


def test_resolve_saelens_config_defaults_for_gpt2_small() -> None:
    config = resolve_saelens_config("gpt2-small", sae_release=None, sae_id=None)

    assert config["release"] == "gpt2-small-res-jb"
    assert config["sae_id"] == "blocks.8.hook_resid_pre"


def test_build_saelens_features_uses_encode_decode_and_effects(tmp_path, monkeypatch) -> None:
    import torch

    class FakeSAE(torch.nn.Module):
        def encode(self, activation):
            return torch.tensor([[[0.0, 3.0, -1.0, 0.5]]], device=activation.device)

        def decode(self, features):
            return features[..., :3]

    class FakeSAEClass:
        @staticmethod
        def from_pretrained(*, release, sae_id, device):
            assert release == "fake-release"
            assert sae_id == "blocks.0.hook_resid_post"
            return FakeSAE().to(device)

    monkeypatch.setitem(
        sys.modules,
        "sae_lens",
        types.SimpleNamespace(SAE=FakeSAEClass),
    )
    activation_path = tmp_path / "activations.npz"
    np.savez_compressed(
        activation_path,
        blocks__0__hook_resid_post=np.array([[[0.0, 3.0, -1.0]]]),
    )

    def effect_fn(replacement):
        return 10.0, float(10.0 + replacement[0, 0, 1].item())

    artifact = build_saelens_features(
        run_id="features",
        backend="transformer-lens",
        behavior=BehaviorSpec(model="fake", prompt="p", target_text=" t"),
        activation_path=activation_path,
        sae_release="fake-release",
        sae_id="blocks.0.hook_resid_post",
        hook_name="blocks.0.hook_resid_post",
        top_k=2,
        feature_ablation=True,
        feature_steering=True,
        steer_scale=2.0,
        effect_fn=effect_fn,
    )

    assert artifact.sae_release == "fake-release"
    assert artifact.sae_feature_dim == 4
    assert len(artifact.features) == 2
    assert artifact.features[0].feature.feature_id == 1
    assert artifact.features[0].ablation_effect is not None
    assert artifact.features[0].steering_effect is not None
    assert artifact.features[0].feature.metadata["metadata_only"] is True
