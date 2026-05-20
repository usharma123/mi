from __future__ import annotations

from mi.core.schema import LocalizationArtifact, LocalizationCandidate, TargetMetrics
from mi.report import (
    render_json_report,
    render_localization_json_report,
    render_localization_markdown,
    render_markdown,
)


def test_markdown_report_contains_required_sections(sample_trace) -> None:
    markdown = render_markdown(sample_trace())

    assert "# Mechanistic Trace Report" in markdown
    assert "## Behavior" in markdown
    assert "## Top Next-Token Predictions" in markdown
    assert "## Logit Lens" in markdown
    assert "## Direct Logit Attribution" in markdown
    assert "## Caveats" in markdown


def test_json_report_contains_trace_summary(sample_trace) -> None:
    report = render_json_report(sample_trace())

    assert report["run_id"] == "sample-run"
    assert report["target"]["token"] == " Paris"
    assert report["top_predictions"][0]["token"] == " Paris"


def test_localization_reports_contain_candidates(sample_trace) -> None:
    trace = sample_trace()
    localization = LocalizationArtifact(
        id="sample-localize",
        backend="transformer-lens",
        behavior=trace.behavior,
        target=TargetMetrics(token_id=6342, token=" Paris", logit=18.4, probability=0.71, rank=1),
        candidates=[
            LocalizationCandidate(
                method="zero_ablation",
                target={"layer": 0, "position": 4, "stream": "resid_post"},
                hook_name="blocks.0.hook_resid_post",
                metric_before=18.4,
                metric_after=16.0,
                effect=2.4,
                rank=1,
            )
        ],
    )

    markdown = render_localization_markdown(localization)
    json_report = render_localization_json_report(localization)

    assert "# Causal Localization Report" in markdown
    assert "zero_ablation" in markdown
    assert json_report["candidates"][0]["effect"] == 2.4
