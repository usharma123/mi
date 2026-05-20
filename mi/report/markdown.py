from __future__ import annotations

from mi.core.schema import TraceArtifact
from mi.methods.direct_logit_attribution import top_direct_attributions


def _cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def render_markdown(trace: TraceArtifact) -> str:
    lines: list[str] = [
        "# Mechanistic Trace Report",
        "",
        "## Behavior",
        "",
        f"- Model: `{trace.behavior.model}`",
        f"- Backend: `{trace.backend}`",
        f"- Prompt: `{trace.behavior.prompt}`",
        f"- Target text: `{trace.behavior.target_text or ''}`",
        f"- Target token: `{trace.behavior.target_token or ''}`",
        "",
        "## Tokenization",
        "",
        "| Position | Token ID | Token |",
        "|---:|---:|---|",
    ]
    for position, (token_id, token) in enumerate(zip(trace.token_ids, trace.tokens)):
        lines.append(f"| {position} | {token_id} | `{_cell(token)}` |")

    lines.extend(["", "## Baseline Metrics", ""])
    if trace.target is None:
        lines.append("No target token metrics were computed.")
    else:
        lines.extend(
            [
                f"- Target logit: `{_float(trace.target.logit)}`",
                f"- Target probability: `{_float(trace.target.probability, 6)}`",
                f"- Target rank: `{trace.target.rank}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Top Next-Token Predictions",
            "",
            "| Rank | Token ID | Token | Logit | Probability |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for item in trace.top_predictions:
        lines.append(
            f"| {item.rank} | {item.token_id} | `{_cell(item.token)}` | "
            f"{_float(item.logit)} | {_float(item.probability, 6)} |"
        )

    lines.extend(
        [
            "",
            "## Logit Lens",
            "",
            "| Layer | Stream | Target Logit | Target Rank | Top Token | Top Logit |",
            "|---:|---|---:|---:|---|---:|",
        ]
    )
    for item in trace.logit_lens:
        lines.append(
            f"| {item.layer} | {item.stream} | {_float(item.target_logit)} | "
            f"{item.target_rank or ''} | `{_cell(item.top_token)}` | {_float(item.top_logit)} |"
        )

    lines.extend(
        [
            "",
            "## Direct Logit Attribution",
            "",
            "| Layer | Component | Position | Contribution |",
            "|---:|---|---:|---:|",
        ]
    )
    attribution_entries = top_direct_attributions(trace.direct_logit_attribution, limit=20)
    if attribution_entries:
        for item in attribution_entries:
            layer = "" if item.layer is None else item.layer
            lines.append(
                f"| {layer} | {item.component} | {item.position} | {_float(item.contribution)} |"
            )
    else:
        lines.append("|  | No target attribution entries |  |  |")

    lines.extend(
        [
            "",
            "## Activation Inventory",
            "",
            f"Captured activation tensors: `{len(trace.activation_inventory)}`",
            "",
            "| Name | Shape | Dtype | Artifact Key |",
            "|---|---|---|---|",
        ]
    )
    for item in trace.activation_inventory[:40]:
        lines.append(
            f"| `{_cell(item.name)}` | `{item.shape}` | `{item.dtype}` | `{item.artifact_key}` |"
        )
    if len(trace.activation_inventory) > 40:
        lines.append(f"| ... | {len(trace.activation_inventory) - 40} more tensors |  |  |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- v0.1 reports traces, associations, and direct attribution estimates; it does not validate causal circuit claims.",
            "- Natural-language labels and feature metadata are not treated as evidence.",
            "- Claims require intervention tests, negative controls, and prompt-family validation in later releases.",
        ]
    )
    if trace.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in trace.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines) + "\n"
