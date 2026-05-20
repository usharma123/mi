from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from mi.cli.main import app
from mi.core.schema import (
    ActivationSummary,
    BehaviorSpec,
    ControlSummary,
    DirectLogitAttributionEntry,
    LocalizationArtifact,
    LocalizationCandidate,
    LogitLensEntry,
    TargetMetrics,
    TopPrediction,
    TraceArtifact,
    ValidationArtifact,
    ValidationResult,
)


class FakeBackend:
    name = "transformer-lens"

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device

    def trace(self, behavior: BehaviorSpec, activation_path, *, run_id: str, top_k: int):
        np.savez_compressed(activation_path, blocks__0__hook_resid_post=np.zeros((1, 3, 2)))
        behavior = behavior.model_copy(update={"target_token": " world"})
        return TraceArtifact(
            id=run_id,
            backend=self.name,
            behavior=behavior,
            token_ids=[1, 2, 3],
            tokens=["Hello", ",", " there"],
            target=TargetMetrics(token_id=10, token=" world", logit=2.0, probability=0.4, rank=1),
            top_predictions=[
                TopPrediction(token_id=10, token=" world", logit=2.0, probability=0.4, rank=1)
            ],
            logit_lens=[
                LogitLensEntry(
                    layer=0,
                    stream="resid_post",
                    target_logit=1.0,
                    target_rank=2,
                    top_token=" world",
                    top_token_id=10,
                    top_logit=2.0,
                )
            ],
            direct_logit_attribution=[
                DirectLogitAttributionEntry(layer=0, component="mlp_out", position=2, contribution=0.5)
            ],
            activation_inventory=[
                ActivationSummary(
                    name="blocks.0.hook_resid_post",
                    shape=[1, 3, 2],
                    dtype="float64",
                    artifact_key="blocks__0__hook_resid_post",
                )
            ],
        )

    def localize(
        self,
        behavior: BehaviorSpec,
        *,
        run_id: str,
        corrupt_prompt: str | None,
        methods: set[str],
        streams: set[str],
        controls: set[str],
        position: str,
        seed: int,
        top_k: int,
    ):
        behavior = behavior.model_copy(update={"target_token": " world"})
        return LocalizationArtifact(
            id=run_id,
            backend=self.name,
            behavior=behavior,
            corrupt_prompt=corrupt_prompt,
            target=TargetMetrics(token_id=10, token=" world", logit=2.0, probability=0.4, rank=1),
            corrupt_target=TargetMetrics(
                token_id=10, token=" world", logit=1.0, probability=0.2, rank=3
            )
            if corrupt_prompt
            else None,
            candidates=[
                LocalizationCandidate(
                    method="zero_ablation",
                    hook_name="blocks.0.hook_resid_post",
                    target={"layer": 0, "position": 2, "stream": "resid_post"},
                    metric_before=2.0,
                    metric_after=1.25,
                    effect=0.75,
                    rank=1,
                    controls=sorted(controls),
                    control_summary=ControlSummary(
                        control_mean=0.1,
                        control_max=0.2,
                        control_count=2,
                        specificity_passed=True,
                        control_effects={"random": [0.1, 0.2]},
                    )
                    if controls
                    else None,
                )
            ][:top_k],
        )


def test_trace_command_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"

    result = runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_path / "manifest.json").exists()
    assert (run_path / "trace.json").exists()
    assert (run_path / "tokens.json").exists()
    assert (run_path / "metrics.json").exists()
    assert (run_path / "activations.npz").exists()
    assert (run_path / "report.md").exists()
    assert (run_path / "report.html").exists()


def test_inspect_and_report_commands(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    inspect_result = runner.invoke(app, ["inspect", str(run_path), "--view", "logits"])
    lens_result = runner.invoke(app, ["inspect", str(run_path), "--view", "logit-lens"])
    report_result = runner.invoke(app, ["report", str(run_path), "--format", "md,json"])

    assert inspect_result.exit_code == 0, inspect_result.output
    assert "Top predictions" in inspect_result.output
    assert lens_result.exit_code == 0, lens_result.output
    assert "target_logit=1.0000" in lens_result.output
    assert report_result.exit_code == 0, report_result.output
    assert (run_path / "report.json").exists()


def test_localize_command_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    trace_result = runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )
    assert trace_result.exit_code == 0, trace_result.output

    result = runner.invoke(
        app,
        [
            "localize",
            str(run_path),
            "--corrupt-prompt",
            "Goodbye, there",
            "--methods",
            "zero-ablation,clean-to-corrupt-patch",
            "--controls",
            "random,same-layer",
            "--position",
            "final",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_path / "localization.json").exists()
    assert (run_path / "candidates.json").exists()
    assert (run_path / "evidence.jsonl").exists()
    assert (run_path / "localize.md").exists()
    assert (run_path / "localize.html").exists()


def test_validate_command_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    claims_path = tmp_path / "claim.yml"
    claims_path.write_text(
        """
id: hello_world
text: Residual stream supports the world token.
target:
  layer: 0
  position: 2
  stream: resid_post
tests:
  - method: zero_ablation
    min_effect: 0.5
    max_control_effect: 0.3
""",
        encoding="utf-8",
    )
    trace_result = runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )
    assert trace_result.exit_code == 0, trace_result.output

    result = runner.invoke(
        app,
        [
            "validate",
            str(run_path),
            "--claims",
            str(claims_path),
            "--controls",
            "random",
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (run_path / "claims.json").exists()
    assert (run_path / "validation.json").exists()
    assert (run_path / "scores.json").exists()
    assert (run_path / "evidence.jsonl").exists()
    assert (run_path / "validate.md").exists()
    assert (run_path / "validate.html").exists()


def test_localize_rejects_invalid_controls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    result = runner.invoke(
        app,
        ["localize", str(run_path), "--controls", "not-a-control"],
    )

    assert result.exit_code == 1
    assert "Unknown control" in result.output


def test_validate_rejects_invalid_thresholds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    claims_path = tmp_path / "claim.yml"
    claims_path.write_text(
        """
id: bad_threshold
text: Bad threshold.
target:
  layer: 0
  position: 2
  stream: resid_post
tests:
  - method: zero_ablation
    min_effect: -1
""",
        encoding="utf-8",
    )
    runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    result = runner.invoke(app, ["validate", str(run_path), "--claims", str(claims_path)])

    assert result.exit_code == 1
    assert "greater than or equal to 0" in result.output


def test_features_command_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    trace_result = runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )
    assert trace_result.exit_code == 0, trace_result.output

    result = runner.invoke(app, ["features", str(run_path), "--top-k", "2"])

    assert result.exit_code == 0, result.output
    assert (run_path / "features.json").exists()
    assert (run_path / "feature_evidence.jsonl").exists()
    assert (run_path / "features.md").exists()
    assert (run_path / "features.html").exists()


def test_graph_command_writes_expected_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    result = runner.invoke(app, ["graph", str(run_path), "--method", "meir"])

    assert result.exit_code == 0, result.output
    assert (run_path / "graph.json").exists()
    assert (run_path / "graph.graphml").exists()
    assert (run_path / "graph.md").exists()
    assert (run_path / "graph.html").exists()


def test_graph_circuit_tracer_import(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_path = tmp_path / "run"
    import_path = tmp_path / "ct.json"
    import_path.write_text(
        '{"nodes":[{"id":"f1","kind":"transcoder_feature"}],"edges":[{"source":"f1","target":"logit","weight":1.0}]}',
        encoding="utf-8",
    )
    runner.invoke(
        app,
        [
            "trace",
            "--model",
            "fake-model",
            "--prompt",
            "Hello, there",
            "--target",
            " world",
            "--out",
            str(run_path),
        ],
    )

    result = runner.invoke(
        app,
        ["graph", str(run_path), "--backend", "circuit-tracer", "--import", str(import_path)],
    )

    assert result.exit_code == 0, result.output
    assert (run_path / "graph.json").exists()


def test_fuzz_command_writes_variants(tmp_path) -> None:
    runner = CliRunner()
    family_path = tmp_path / "family.yml"
    out = tmp_path / "variants.jsonl"
    family_path.write_text(
        """
family: test
templates:
  - "Hello {name}"
variables:
  name:
    - Ada
    - Grace
target_template: " world"
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["fuzz", str(family_path), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_backends_command_lists_capabilities() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["backends"])

    assert result.exit_code == 0, result.output
    assert "transformer-lens" in result.output
    assert "circuit-tracer" in result.output


def test_diff_command_writes_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    out = tmp_path / "diff"
    for run_path, model in ((run_a, "fake-a"), (run_b, "fake-b")):
        result = runner.invoke(
            app,
            [
                "trace",
                "--model",
                model,
                "--prompt",
                "Hello, there",
                "--target",
                " world",
                "--out",
                str(run_path),
            ],
        )
        assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["diff", "--run-a", str(run_a), "--run-b", str(run_b), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "diff.json").exists()
    assert (out / "diff.md").exists()
    assert (out / "diff.html").exists()


def test_test_command_loads_claims_and_returns_untested(tmp_path) -> None:
    runner = CliRunner()
    claim = tmp_path / "claim.yml"
    claim.write_text(
        "id: test_claim\ntext: Test claim.\ntarget:\n  layer: 0\n  position: 0\n  stream: resid_post\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["test", str(claim), "--model", "fake"])

    assert result.exit_code == 3
    assert "test_claim -> untested" in result.output


def test_test_command_executes_claim_with_behavior(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("mi.cli.main.get_backend", lambda name: FakeBackend)
    runner = CliRunner()
    claim = tmp_path / "claim.yml"
    claim.write_text(
        "\n".join(
            [
                "id: executable_claim",
                "text: Executable claim.",
                "target:",
                "  layer: 0",
                "  position: 2",
                "  stream: resid_post",
                "hook_name: blocks.0.hook_resid_post",
                "behavior:",
                "  model: fake-model",
                "  prompt: Hello, there",
                "  target_text: ' world'",
                "tests:",
                "  - method: zero_ablation",
                "    min_effect: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["test", str(claim), "--device", "cpu"])

    assert result.exit_code == 0, result.output
    assert "executable_claim -> supported" in result.output


def test_test_command_enforces_run_validation_artifact(tmp_path) -> None:
    runner = CliRunner()
    claim = tmp_path / "claim.yml"
    claim.write_text(
        "id: test_claim\ntext: Test claim.\ntarget:\n  layer: 0\n  position: 0\n  stream: resid_post\n",
        encoding="utf-8",
    )
    run_path = tmp_path / "run"
    run_path.mkdir()
    validation = ValidationArtifact(
        id="validation",
        backend="transformer-lens",
        results=[ValidationResult(claim_id="test_claim", verdict="supported")],
    )
    (run_path / "validation.json").write_text(validation.model_dump_json(), encoding="utf-8")

    result = runner.invoke(app, ["test", str(claim), "--run", str(run_path), "--model", "fake"])

    assert result.exit_code == 0, result.output
    assert "test_claim -> supported" in result.output


def test_test_command_marks_missing_validation_claim_untested(tmp_path) -> None:
    runner = CliRunner()
    claim = tmp_path / "claim.yml"
    claim.write_text(
        "id: missing_claim\ntext: Missing claim.\ntarget:\n  layer: 0\n  position: 0\n  stream: resid_post\n",
        encoding="utf-8",
    )
    validation = ValidationArtifact(
        id="validation",
        backend="transformer-lens",
        results=[ValidationResult(claim_id="other_claim", verdict="supported")],
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(validation.model_dump_json(), encoding="utf-8")

    result = runner.invoke(app, ["test", str(claim), "--validation", str(validation_path)])

    assert result.exit_code == 3
    assert "missing_claim -> untested" in result.output
    assert "did not include requested claim" in result.output
