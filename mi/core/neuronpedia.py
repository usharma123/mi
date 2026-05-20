from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_neuronpedia_metadata(
    *,
    model: str,
    sae_release: str,
    sae_id: str,
    feature_ids: list[int],
    cache_dir: Path,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    missing: list[int] = []
    for feature_id in feature_ids:
        cache_path = _cache_path(cache_dir, model, sae_release, sae_id, feature_id)
        if cache_path.exists():
            try:
                metadata[feature_id] = json.loads(cache_path.read_text(encoding="utf-8"))
                continue
            except Exception as exc:
                warnings.append(f"Neuronpedia cache read failed for feature {feature_id}: {exc}")
        missing.append(feature_id)

    if not missing:
        return metadata, warnings

    try:
        import neuronpedia
    except Exception as exc:
        warnings.append(f"Neuronpedia metadata unavailable: {exc}")
        return metadata, warnings

    for feature_id in missing:
        try:
            payload = _fetch_feature(neuronpedia, model, sae_release, sae_id, feature_id)
            if payload:
                payload = {**payload, "metadata_only": True, "source": "neuronpedia"}
                metadata[feature_id] = payload
                cache_path = _cache_path(cache_dir, model, sae_release, sae_id, feature_id)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                warnings.append(f"Neuronpedia lookup returned no metadata for feature {feature_id}.")
        except Exception as exc:
            warnings.append(f"Neuronpedia lookup failed for feature {feature_id}: {exc}")
    return metadata, warnings


def _fetch_feature(neuronpedia, model: str, sae_release: str, sae_id: str, feature_id: int) -> dict[str, Any]:
    source = _source_from_sae_id(sae_id)
    if hasattr(neuronpedia, "FeatureRequest") and hasattr(neuronpedia, "np_feature"):
        request = neuronpedia.FeatureRequest(modelId=model, source=source, index=str(feature_id))
        response = neuronpedia.np_feature(request)
    elif hasattr(neuronpedia, "SAEFeatureRequest") and hasattr(neuronpedia, "np_sae_feature"):
        request = neuronpedia.SAEFeatureRequest(modelId=model, source=source, index=str(feature_id))
        response = neuronpedia.np_sae_feature(request)
    else:
        return {}
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {
            key: value
            for key in dir(response)
            if not key.startswith("_")
            and isinstance((value := getattr(response, key)), (str, int, float, list, dict, type(None)))
        }
    label = payload.get("label") or payload.get("explanation") or payload.get("description")
    if label:
        payload["label"] = label
    payload.setdefault("neuronpedia_source", source)
    payload.setdefault("sae_release", sae_release)
    return payload


def _source_from_sae_id(sae_id: str) -> str:
    return sae_id.replace("blocks.", "").replace(".hook_", "-").replace("/", "-")


def _cache_path(cache_dir: Path, model: str, sae_release: str, sae_id: str, feature_id: int) -> Path:
    safe = "__".join([model, sae_release, sae_id, str(feature_id)])
    safe = safe.replace("/", "_").replace(".", "_").replace(":", "_")
    return cache_dir / f"{safe}.json"
