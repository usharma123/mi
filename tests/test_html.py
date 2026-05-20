from __future__ import annotations

from mi.report import render_html


def test_render_html_contains_sortable_table_script() -> None:
    html = render_html("Title", "# Title\n\n| A | B |\n|---|---|\n| 2 | x |\n")

    assert "<html" in html
    assert "sortTable" in html
    assert "<table>" in html
