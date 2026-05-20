from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from mi.cli.main import app
from mi.core.schema import (
    ActivationSummary,
    BehaviorSpec,
    DirectLogitAttributionEntry,
    LogitLensEntry,
    TargetMetrics,
    TopPrediction,
    TraceArtifact,
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
