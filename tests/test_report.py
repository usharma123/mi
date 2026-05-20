from __future__ import annotations

from mi.report import render_json_report, render_markdown


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
