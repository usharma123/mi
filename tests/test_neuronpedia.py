from __future__ import annotations

import sys
import types

from mi.core.neuronpedia import load_neuronpedia_metadata


def test_neuronpedia_metadata_cache_hit(tmp_path) -> None:
    cache_file = tmp_path / "gpt2-small__release__sae__1.json"
    cache_file.write_text('{"label": "cached", "metadata_only": true}\n', encoding="utf-8")

    metadata, warnings = load_neuronpedia_metadata(
        model="gpt2-small",
        sae_release="release",
        sae_id="sae",
        feature_ids=[1],
        cache_dir=tmp_path,
    )

    assert metadata[1]["label"] == "cached"
    assert warnings == []


def test_neuronpedia_metadata_failure_is_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "neuronpedia", types.SimpleNamespace())

    metadata, warnings = load_neuronpedia_metadata(
        model="gpt2-small",
        sae_release="release",
        sae_id="sae",
        feature_ids=[2],
        cache_dir=tmp_path,
    )

    assert metadata == {}
    assert warnings
