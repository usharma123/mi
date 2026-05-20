from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from mi.core.schema import ActivationRef, Evidence, FeatureArtifact, FeatureCandidate, FeatureRef
from mi.methods.activation_capture import activation_artifact_key


DEFAULT_SAELENS_BY_MODEL = {
    "gpt2-small": {
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.8.hook_resid_pre",
        "hook_name": "blocks.8.hook_resid_pre",
    }
}


def resolve_saelens_config(
    model: str,
    *,
    sae_release: str | None,
    sae_id: str | None,
) -> dict[str, str]:
    if sae_release and sae_id:
        hook_name = sae_id if sae_id.startswith("blocks.") else DEFAULT_SAELENS_BY_MODEL.get(model, {}).get("hook_name", sae_id)
        return {"release": sae_release, "sae_id": sae_id, "hook_name": hook_name}
    defaults = DEFAULT_SAELENS_BY_MODEL.get(model)
    if defaults is None:
        raise ValueError("--sae-release and --sae-id are required for models without mi defaults.")
    return {
        "release": sae_release or defaults["release"],
        "sae_id": sae_id or defaults["sae_id"],
        "hook_name": defaults["hook_name"],
    }


def build_saelens_features(
    *,
    run_id: str,
    backend: str,
    behavior,
    activation_path: Path,
    sae_release: str,
    sae_id: str,
    hook_name: str,
    top_k: int,
    device: str | None = None,
    metadata: dict[int, dict[str, Any]] | None = None,
    metadata_loader: Callable[[list[int]], tuple[dict[int, dict[str, Any]], list[str]]] | None = None,
    feature_ablation: bool = False,
    feature_steering: bool = False,
    steer_scale: float = 2.0,
    effect_fn: Callable[[Any], tuple[float, float]] | None = None,
) -> FeatureArtifact:
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation artifact not found: {activation_path}")

    try:
        import torch
        from sae_lens import SAE
    except ImportError as exc:
        raise RuntimeError("SAELens is required for --dictionary saelens.") from exc

    loaded = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device or "cpu")
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    if hasattr(sae, "eval"):
        sae.eval()

    artifact_key = activation_artifact_key(hook_name)
    arrays = np.load(activation_path)
    if artifact_key not in arrays:
        raise ValueError(
            f"Activation `{hook_name}` was not captured in activations.npz. "
            "Run `mi trace` with a compatible TransformerLens model/hook."
        )

    parsed = _parse_activation_key(artifact_key)
    if parsed is None:
        raise ValueError(f"Unsupported SAE hook for feature extraction: {hook_name}")
    layer, stream = parsed
    array = arrays[artifact_key]
    position = int(array.shape[1] - 1)
    activation = torch.as_tensor(array[:, position : position + 1, :], dtype=torch.float32)
    if hasattr(sae, "parameters"):
        first_param = next(sae.parameters(), None)
        sae_device = first_param.device if first_param is not None else torch.device(device or "cpu")
    else:
        sae_device = torch.device(device or "cpu")
    activation = activation.to(sae_device)

    with torch.no_grad():
        encoded = sae.encode(activation)
        decoded = sae.decode(encoded)
        reconstruction_error = float(
            torch.linalg.vector_norm((decoded - activation).float()).item()
            / max(torch.linalg.vector_norm(activation.float()).item(), 1e-12)
        )
        vector = encoded[0, -1].detach()
        limit = min(max(top_k, 1), int(vector.numel()))
        values, indices = torch.topk(torch.abs(vector), k=limit)
        feature_ids = [int(index) for index in indices.tolist()]
        warnings: list[str] = []
        if metadata_loader is not None:
            loaded_metadata, metadata_warnings = metadata_loader(feature_ids)
            metadata = {**(metadata or {}), **loaded_metadata}
            warnings.extend(metadata_warnings)

        candidates: list[FeatureCandidate] = []
        evidence: list[Evidence] = []
        for activation_abs, feature_id_tensor in zip(values.tolist(), indices.tolist()):
            feature_id = int(feature_id_tensor)
            activation_value = float(vector[feature_id].item())
            metadata_payload = {
                "metadata_only": True,
                "source": "saelens",
                "sae_release": sae_release,
                "sae_id": sae_id,
                "hook_name": hook_name,
                "activation_abs": float(activation_abs),
            }
            if metadata and feature_id in metadata:
                metadata_payload.update(metadata[feature_id])
            label = metadata_payload.get("label")
            feature_ref = FeatureRef(
                dictionary_id=f"saelens/{sae_release}/{sae_id}",
                layer=layer,
                feature_id=feature_id,
                source="sae",
                label=str(label) if label else f"{sae_id} feature {feature_id}",
                metadata=metadata_payload,
            )
            activation_ref = ActivationRef(layer=layer, position=position, stream=stream)
            evidence_ids: list[str] = []
            evidence_id = f"feature_ev_{len(evidence) + 1}"
            evidence.append(
                Evidence(
                    id=evidence_id,
                    method="sae_feature_activation",
                    target=feature_ref,
                    metric_before=0.0,
                    metric_after=activation_value,
                    delta=activation_value,
                    artifact_refs=["features.json"],
                )
            )
            evidence_ids.append(evidence_id)

            ablation_effect = None
            steering_effect = None
            if effect_fn and feature_ablation:
                ablated = encoded.clone()
                ablated[..., feature_id] = 0
                replacement = sae.decode(ablated)
                before, after = effect_fn(replacement.detach())
                ablation_effect = before - after
                evidence_id = f"feature_ev_{len(evidence) + 1}"
                evidence.append(
                    Evidence(
                        id=evidence_id,
                        method="feature_ablation",
                        target=feature_ref,
                        metric_before=before,
                        metric_after=after,
                        delta=after - before,
                        artifact_refs=["features.json"],
                    )
                )
                evidence_ids.append(evidence_id)
            if effect_fn and feature_steering:
                steered = encoded.clone()
                steered[..., feature_id] = steered[..., feature_id] * steer_scale
                replacement = sae.decode(steered)
                before, after = effect_fn(replacement.detach())
                steering_effect = after - before
                evidence_id = f"feature_ev_{len(evidence) + 1}"
                evidence.append(
                    Evidence(
                        id=evidence_id,
                        method="feature_steering",
                        target=feature_ref,
                        metric_before=before,
                        metric_after=after,
                        delta=after - before,
                        artifact_refs=["features.json"],
                    )
                )
                evidence_ids.append(evidence_id)

            candidates.append(
                FeatureCandidate(
                    feature=feature_ref,
                    activation_ref=activation_ref,
                    activation_value=activation_value,
                    dictionary_reconstruction_error=reconstruction_error,
                    ablation_effect=ablation_effect,
                    steering_effect=steering_effect,
                    evidence_ids=evidence_ids,
                )
            )

    return FeatureArtifact(
        id=run_id,
        backend=backend,
        behavior=behavior,
        dictionary_id=f"saelens/{sae_release}/{sae_id}",
        source="saelens",
        sae_release=sae_release,
        sae_id=sae_id,
        hook_name=hook_name,
        sae_input_dim=int(activation.shape[-1]),
        sae_feature_dim=int(encoded.shape[-1]),
        features=candidates,
        evidence=evidence,
        warnings=warnings,
    )


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
