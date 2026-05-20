from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mi import __version__
from mi.backends import get_backend
from mi.core.artifact_store import ArtifactStore, default_run_dir
from mi.core.schema import BehaviorSpec, LocalizationArtifact, RunManifest, TraceArtifact
from mi.report import (
    render_json_report,
    render_localization_markdown,
    render_markdown,
)

app = typer.Typer(
    add_completion=False,
    help="Mechanistic interpretability traces as reproducible evidence artifacts.",
)


def _load_trace(run_path: Path) -> tuple[ArtifactStore, TraceArtifact]:
    store = ArtifactStore(run_path)
    trace_path = store.path("trace.json")
    if not trace_path.exists():
        raise typer.BadParameter(f"No trace.json found in {run_path}")
    return store, store.read_model("trace.json", TraceArtifact)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the mi version and exit."),
    ] = False,
) -> None:
    if version:
        typer.echo(f"mi {__version__}")
        raise typer.Exit()


@app.command("trace")
def trace_command(
    model: Annotated[str, typer.Option("--model", help="Model name to load.")] = "gpt2-small",
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt to trace.")] = ...,
    target: Annotated[str, typer.Option("--target", "-t", help="Target next-token text.")] = ...,
    backend: Annotated[
        str,
        typer.Option("--backend", help="Backend adapter to use."),
    ] = "transformer-lens",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output run directory."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device to use: auto, cpu, cuda, or mps."),
    ] = "auto",
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Number of next-token predictions to keep."),
    ] = 10,
) -> None:
    run_path = out or default_run_dir(prompt)
    store = ArtifactStore.create(run_path)
    behavior = BehaviorSpec(model=model, prompt=prompt, target_text=target)

    try:
        backend_factory = get_backend(backend)
        trace_backend = backend_factory(model, device)
        trace = trace_backend.trace(
            behavior,
            store.path("activations.npz"),
            run_id=store.run_id,
            top_k=top_k,
        )
    except Exception as exc:
        typer.secho(f"trace failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    artifact_refs = {
        "manifest": "manifest.json",
        "trace": "trace.json",
        "tokens": "tokens.json",
        "metrics": "metrics.json",
        "activations": "activations.npz",
        "report_md": "report.md",
    }
    trace = trace.model_copy(update={"artifact_refs": artifact_refs})

    store.write_json("trace.json", trace)
    store.write_json(
        "tokens.json",
        {
            "token_ids": trace.token_ids,
            "tokens": trace.tokens,
            "target_token": trace.behavior.target_token,
        },
    )
    store.write_json(
        "metrics.json",
        {
            "target": trace.target.model_dump(mode="json") if trace.target else None,
            "top_predictions": [item.model_dump(mode="json") for item in trace.top_predictions],
            "logit_lens": [item.model_dump(mode="json") for item in trace.logit_lens],
            "direct_logit_attribution": [
                item.model_dump(mode="json") for item in trace.direct_logit_attribution
            ],
        },
    )
    store.write_text("report.md", render_markdown(trace))
    store.write_manifest(
        RunManifest(
            run_id=trace.id,
            version=__version__,
            backend=trace.backend,
            behavior=trace.behavior,
            artifacts=artifact_refs,
        )
    )

    typer.echo(f"Wrote trace artifacts to {store.root}")


@app.command("inspect")
def inspect_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    view: Annotated[
        str,
        typer.Option("--view", help="View to print: logits, logit-lens, or activations."),
    ] = "logits",
) -> None:
    _, trace = _load_trace(run_path)
    normalized = view.lower()

    if normalized == "logits":
        if trace.target:
            typer.echo(
                f"Target {trace.target.token!r}: logit={trace.target.logit:.4f} "
                f"prob={trace.target.probability:.6f} rank={trace.target.rank}"
            )
        else:
            typer.echo("No target metrics were computed.")
        typer.echo("Top predictions:")
        for item in trace.top_predictions:
            typer.echo(
                f"{item.rank:>2}. {item.token!r:<16} id={item.token_id:<8} "
                f"logit={item.logit:.4f} prob={item.probability:.6f}"
            )
        return

    if normalized in {"logit-lens", "logitlens"}:
        for item in trace.logit_lens:
            target = "" if item.target_logit is None else f" target_logit={item.target_logit:.4f}"
            rank = "" if item.target_rank is None else f" target_rank={item.target_rank}"
            top_logit = "" if item.top_logit is None else f"{item.top_logit:.4f}"
            typer.echo(
                f"L{item.layer:<3} {item.stream:<10} top={item.top_token!r} "
                f"top_logit={top_logit}{target}{rank}"
            )
        return

    if normalized == "activations":
        for item in trace.activation_inventory:
            typer.echo(f"{item.artifact_key}: {item.name} shape={item.shape} dtype={item.dtype}")
        return

    typer.secho(f"Unknown inspect view: {view}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command("localize")
def localize_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    corrupt_prompt: Annotated[
        str | None,
        typer.Option("--corrupt-prompt", help="Optional corrupt prompt for clean-to-corrupt patching."),
    ] = None,
    methods: Annotated[
        str,
        typer.Option(
            "--methods",
            help="Comma-separated localization methods: zero-ablation,clean-to-corrupt-patch.",
        ),
    ] = "zero-ablation",
    streams: Annotated[
        str,
        typer.Option("--streams", help="Comma-separated streams: resid_post,attn_out,mlp_out."),
    ] = "resid_post,attn_out,mlp_out",
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Number of ranked candidates to keep."),
    ] = 20,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Backend adapter override. Defaults to trace backend."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device to use: auto, cpu, cuda, or mps."),
    ] = "auto",
) -> None:
    store, trace = _load_trace(run_path)
    method_set = {
        item.strip().lower().replace("-", "_") for item in methods.split(",") if item.strip()
    }
    stream_set = {item.strip().lower() for item in streams.split(",") if item.strip()}
    supported_methods = {"zero_ablation", "clean_to_corrupt_patch"}
    unknown_methods = method_set - supported_methods
    if unknown_methods:
        typer.secho(
            f"Unknown localization method(s): {', '.join(sorted(unknown_methods))}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    selected_backend = backend or trace.backend
    try:
        backend_factory = get_backend(selected_backend)
        trace_backend = backend_factory(trace.behavior.model, device)
        localization = trace_backend.localize(
            trace.behavior,
            run_id=f"{store.run_id}-localize",
            corrupt_prompt=corrupt_prompt,
            methods=method_set,
            streams=stream_set,
            top_k=top_k,
        )
    except Exception as exc:
        typer.secho(f"localize failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    artifact_refs = {
        "localization": "localization.json",
        "candidates": "candidates.json",
        "evidence": "evidence.jsonl",
        "report_md": "localize.md",
    }
    localization = localization.model_copy(update={"artifact_refs": artifact_refs})
    store.write_json("localization.json", localization)
    store.write_json(
        "candidates.json",
        [candidate.model_dump(mode="json") for candidate in localization.candidates],
    )
    evidence_lines = "\n".join(
        evidence.model_dump_json() for evidence in localization.evidence
    )
    store.write_text("evidence.jsonl", evidence_lines + ("\n" if evidence_lines else ""))
    store.write_text("localize.md", render_localization_markdown(localization))
    typer.echo(f"Wrote localization artifacts to {store.root}")


@app.command("report")
def report_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    output_format: Annotated[
        str,
        typer.Option("--format", help="Comma-separated formats to write: md,json."),
    ] = "md",
) -> None:
    store, trace = _load_trace(run_path)
    formats = {item.strip().lower() for item in output_format.split(",") if item.strip()}
    unknown = formats - {"md", "json"}
    if unknown:
        typer.secho(f"Unknown report format(s): {', '.join(sorted(unknown))}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    written: list[Path] = []
    if "md" in formats:
        written.append(store.write_text("report.md", render_markdown(trace)))
    if "json" in formats:
        written.append(store.write_json("report.json", render_json_report(trace)))

    for path in written:
        typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
