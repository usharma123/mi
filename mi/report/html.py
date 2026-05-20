from __future__ import annotations

import html


def render_html(title: str, markdown: str) -> str:
    body = _markdownish_to_html(markdown)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f6f6; cursor: pointer; }}
    code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 4px; }}
    .note {{ color: #555; }}
  </style>
  <script>
    function sortTable(table, col) {{
      const rows = Array.from(table.tBodies[0].rows);
      rows.sort((a, b) => a.cells[col].innerText.localeCompare(b.cells[col].innerText, undefined, {{numeric: true}}));
      rows.forEach(row => table.tBodies[0].appendChild(row));
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      document.querySelectorAll('table').forEach(table => {{
        table.querySelectorAll('th').forEach((th, i) => th.addEventListener('click', () => sortTable(table, i)));
      }});
    }});
  </script>
</head>
<body>
{body}
</body>
</html>
"""


def _markdownish_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    html_lines = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            html_lines.append(f"<p class=\"note\">{html.escape(line[2:])}</p>")
        elif line.startswith("|") and line.endswith("|"):
            cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            if set(cells) == {"---"} or all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                html_lines.append("<table><thead><tr>")
                html_lines.extend(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells)
                html_lines.append("</tr></thead><tbody>")
                in_table = True
            else:
                html_lines.append("<tr>")
                html_lines.extend(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells)
                html_lines.append("</tr>")
        else:
            if in_table:
                html_lines.append("</tbody></table>")
                in_table = False
            if line.strip():
                html_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        html_lines.append("</tbody></table>")
    return "\n".join(html_lines)
