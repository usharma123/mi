from __future__ import annotations

from mi import __version__
from mi.core.artifact_store import ArtifactStore, default_run_dir, slugify
from mi.core.schema import BehaviorSpec, RunManifest


def test_slugify_and_default_run_dir() -> None:
    assert slugify("The capital of France is") == "the-capital-of-france-is"
    assert default_run_dir("The capital of France is").parent.name == "runs"


def test_artifact_store_writes_and_reads_manifest(tmp_path) -> None:
    store = ArtifactStore.create(tmp_path / "run")
    behavior = BehaviorSpec(model="gpt2-small", prompt="Hello", target_text=" world")
    manifest = RunManifest(
        run_id=store.run_id,
        version=__version__,
        backend="transformer-lens",
        behavior=behavior,
        artifacts={"trace": "trace.json"},
    )

    store.write_manifest(manifest)
    restored = store.read_manifest()

    assert restored == manifest
    assert store.relative_ref(store.path("trace.json")) == "trace.json"
