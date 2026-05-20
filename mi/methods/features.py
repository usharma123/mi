from __future__ import annotations

from pathlib import Path

import numpy as np

from mi.core.schema import ActivationRef, Evidence, FeatureArtifact, FeatureCandidate, FeatureRef


def build_raw_activation_features(
    *,
    run_id: str,
    backend: str,
    behavior,
    activation_path: Path,
    dictionary_id: str,
    source: str,
    top_k: int,
    layer: int | None = None,
    stream: str | None = None,
) -> FeatureArtifact:
    warnings: list[str] = []
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation artifact not found: {activation_path}")

    arrays = np.load(activation_path)
    candidates: list[FeatureCandidate] = []
    evidence: list[Evidence] = []
    for key in sorted(arrays.files):
        parsed = _parse_activation_key(key)
        if parsed is None:
            continue
        parsed_layer, parsed_stream = parsed
        if layer is not None and parsed_layer != layer:
            continue
        if stream is not None and parsed_stream != stream:
            continue

        array = arrays[key]
        if array.ndim < 3:
            continue
        position = int(array.shape[1] - 1)
        vector = np.asarray(array[0, position], dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            continue
        ranked_indices = np.argsort(np.abs(vector))[::-1][: max(top_k, 1)]
        for feature_id in ranked_indices.tolist():
            activation_value = float(vector[int(feature_id)])
            feature_ref = FeatureRef(
                dictionary_id=dictionary_id,
                layer=parsed_layer,
                feature_id=int(feature_id),
                source="custom_direction" if source == "raw_activation" else "sae",
                label=f"{parsed_stream} dimension {feature_id}",
                metadata={
                    "activation_key": key,
                    "metadata_only": True,
                    "source": source,
                },
            )
            activation_ref = ActivationRef(
                layer=parsed_layer,
                position=position,
                stream=parsed_stream,
            )
            evidence_id = f"feature_ev_{len(evidence) + 1}"
            evidence.append(
                Evidence(
                    id=evidence_id,
                    method="feature_activation",
                    target=feature_ref,
                    metric_before=0.0,
                    metric_after=activation_value,
                    delta=activation_value,
                    artifact_refs=["features.json"],
                )
            )
            candidates.append(
                FeatureCandidate(
                    feature=feature_ref,
                    activation_ref=activation_ref,
                    activation_value=activation_value,
                    dictionary_reconstruction_error=None,
                    evidence_ids=[evidence_id],
                )
            )

    candidates = sorted(candidates, key=lambda item: abs(item.activation_value), reverse=True)[:top_k]
    if source == "saelens":
        warnings.append(
            "SAELens-compatible feature artifact generated from cached activations; "
            "no pretrained SAE release was supplied, so labels are raw metadata."
        )
    return FeatureArtifact(
        id=run_id,
        backend=backend,
        behavior=behavior,
        dictionary_id=dictionary_id,
        source=source,
        features=candidates,
        evidence=evidence,
        warnings=warnings,
    )


def _parse_activation_key(key: str) -> tuple[int, str] | None:
    parts = key.split("__")
    if len(parts) < 3 or parts[0] != "blocks":
        return None
    try:
        layer = int(parts[1])
    except ValueError:
        return None
    hook = "__".join(parts[2:])
    stream_by_hook = {
        "hook_resid_pre": "resid_pre",
        "hook_resid_mid": "resid_mid",
        "hook_resid_post": "resid_post",
        "hook_attn_out": "attn_out",
        "hook_mlp_out": "mlp_out",
    }
    stream = stream_by_hook.get(hook)
    if stream is None:
        return None
    return layer, stream
