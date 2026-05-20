from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from mi import __version__
from mi.backends import get_backend
from mi.core.artifact_store import ArtifactStore, default_run_dir
from mi.core.circuit_tracer import import_circuit_tracer_graph
from mi.core.cts import score_validation
from mi.core.diff import diff_traces
from mi.core.fuzz import generate_variants, load_prompt_family, load_variants_jsonl, write_variants_jsonl
from mi.core.schema import (
    BehaviorSpec,
    FeatureArtifact,
    GraphArtifact,
    LocalizationArtifact,
    RunManifest,
    TraceArtifact,
    ValidationArtifact,
)
from mi.core.graph_builder import build_meir_graph, graph_to_graphml
from mi.core.validation import evaluate_claim, find_candidate, load_claim_specs
from mi.methods.features import build_raw_activation_features
from mi.methods.localization import parse_controls
from mi.report import (
    render_features_markdown,
    render_diff_markdown,
    render_graph_markdown,
    render_html,
    render_json_report,
    render_localization_markdown,
    render_markdown,
    render_validation_markdown,
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
        "report_html": "report.html",
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
    store.write_text("report.html", render_html("Mechanistic Trace Report", render_markdown(trace)))
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


@app.command("backends")
def backends_command() -> None:
    for name in ("transformer-lens", "nnsight", "hf-hooks", "circuit-tracer"):
        factory = get_backend(name)
        backend = factory("<model>", None)
        caps = backend.capabilities() if hasattr(backend, "capabilities") else {}
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(caps.items()))
        typer.echo(f"{name}: {rendered}")


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
    controls: Annotated[
        str,
        typer.Option(
            "--controls",
            help="Comma-separated controls: random,same-layer,wrong-target,wrong-corrupt.",
        ),
    ] = "",
    position: Annotated[
        str,
        typer.Option("--position", help="Position spec: final, first-diff, index:N, or token:TEXT."),
    ] = "final",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for deterministic control sampling."),
    ] = 0,
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
    try:
        control_set = parse_controls(controls)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

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
            controls=control_set,
            position=position,
            seed=seed,
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
        "report_html": "localize.html",
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
    store.write_text("localize.html", render_html("Causal Localization Report", render_localization_markdown(localization)))
    typer.echo(f"Wrote localization artifacts to {store.root}")


@app.command("validate")
def validate_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    claims_path: Annotated[
        Path,
        typer.Option("--claims", help="YAML or JSON claim spec file."),
    ],
    controls: Annotated[
        str,
        typer.Option(
            "--controls",
            help="Comma-separated controls: random,same-layer,wrong-target,wrong-corrupt.",
        ),
    ] = "",
    variants_path: Annotated[
        Path | None,
        typer.Option("--variants", help="Optional variants JSONL file for robustness metadata."),
    ] = None,
    position: Annotated[
        str,
        typer.Option("--position", help="Position spec for rerun localization."),
    ] = "final",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random seed for deterministic control sampling."),
    ] = 0,
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
    try:
        claims = load_claim_specs(claims_path)
        control_set = parse_controls(controls)
        variants = load_variants_jsonl(variants_path) if variants_path else []
    except Exception as exc:
        typer.secho(f"validate failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    selected_backend = backend or trace.backend
    localizations: list[LocalizationArtifact] = []
    if store.path("localization.json").exists():
        localizations.append(store.read_model("localization.json", LocalizationArtifact))

    backend_instances = {}
    warnings: list[str] = []
    validation_evidence = []
    validation_results = []
    for claim in claims:
        claim_tests = claim.tests
        behavior = claim.behavior or trace.behavior
        corrupt_prompt = claim.corrupt_prompt
        resolved_tests = []
        for test in claim_tests:
            candidate = _find_candidate_in_localizations(localizations, claim, test)
            needs_controls = control_set and (
                candidate is None or candidate.control_summary is None
            )
            if candidate is None or needs_controls:
                backend_key = (selected_backend, behavior.model, device)
                if backend_key not in backend_instances:
                    try:
                        backend_factory = get_backend(selected_backend)
                        backend_instances[backend_key] = backend_factory(behavior.model, device)
                    except Exception as exc:
                        typer.secho(f"validate failed: {exc}", fg=typer.colors.RED, err=True)
                        raise typer.Exit(code=1) from exc
                try:
                    rerun = backend_instances[backend_key].localize(
                        behavior,
                        run_id=f"{store.run_id}-validate-{claim.id}",
                        corrupt_prompt=corrupt_prompt,
                        methods={test.method},
                        streams={claim.target.stream},
                        controls=control_set,
                        position=position,
                        seed=seed,
                        top_k=999,
                    )
                except Exception as exc:
                    typer.secho(f"validate failed: {exc}", fg=typer.colors.RED, err=True)
                    raise typer.Exit(code=1) from exc
                localizations.append(rerun)
                warnings.extend(rerun.warnings)
                candidate = _find_candidate_in_localizations([rerun], claim, test)
            resolved_tests.append((test, candidate))

        result, evidence = evaluate_claim(
            claim,
            resolved_tests,
            evidence_start=len(validation_evidence) + 1,
        )
        validation_results.append(result)
        validation_evidence.extend(evidence)

    artifact_refs = {
        "claims": "claims.json",
        "validation": "validation.json",
        "scores": "scores.json",
        "evidence": "evidence.jsonl",
        "report_md": "validate.md",
        "report_html": "validate.html",
    }
    validation = ValidationArtifact(
        id=f"{store.run_id}-validate",
        backend=selected_backend,
        claims=claims,
        results=validation_results,
        evidence=validation_evidence,
        artifact_refs=artifact_refs,
        warnings=warnings,
    )
    scores = score_validation(validation)
    if variants:
        validation.warnings.append(
            f"Loaded {len(variants)} prompt variants; full rerun pass-rate validation is scheduled for backend-specific suites."
        )
    store.write_json("claims.json", [claim.model_dump(mode="json") for claim in claims])
    store.write_json("validation.json", validation)
    store.write_json("scores.json", scores)
    evidence_lines = "\n".join(
        evidence.model_dump_json() for evidence in validation.evidence
    )
    store.write_text("evidence.jsonl", evidence_lines + ("\n" if evidence_lines else ""))
    store.write_text("validate.md", render_validation_markdown(validation))
    store.write_text("validate.html", render_html("Claim Validation Report", render_validation_markdown(validation)))
    typer.echo(f"Wrote validation artifacts to {store.root}")


@app.command("fuzz")
def fuzz_command(
    family_path: Annotated[Path, typer.Argument(help="Prompt-family YAML file.")],
    out: Annotated[Path, typer.Option("--out", help="Output variants JSONL path.")],
) -> None:
    try:
        family = load_prompt_family(family_path)
        variants = generate_variants(family)
        write_variants_jsonl(out, variants)
    except Exception as exc:
        typer.secho(f"fuzz failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {len(variants)} variants to {out}")


def _find_candidate_in_localizations(localizations, claim, test):
    for localization in localizations:
        candidate = find_candidate(localization, claim, test)
        if candidate is not None:
            return candidate
    return None


@app.command("features")
def features_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    dictionary: Annotated[
        str,
        typer.Option("--dictionary", help="Dictionary source: saelens, raw-activation, or custom."),
    ] = "raw-activation",
    source: Annotated[
        str | None,
        typer.Option("--source", help="Optional metadata source such as neuronpedia."),
    ] = None,
    top_k: Annotated[
        int,
        typer.Option("--top-k", min=1, help="Number of features to keep."),
    ] = 50,
    layer: Annotated[
        int | None,
        typer.Option("--layer", help="Optional layer filter."),
    ] = None,
    stream: Annotated[
        str | None,
        typer.Option("--stream", help="Optional stream filter such as resid_post or mlp_out."),
    ] = None,
) -> None:
    store, trace = _load_trace(run_path)
    normalized = dictionary.strip().lower().replace("_", "-")
    source_name = "raw_activation" if normalized in {"raw", "raw-activation"} else normalized
    dictionary_id = normalized
    if normalized == "saelens":
        dictionary_id = f"saelens/{trace.behavior.model}"
    elif normalized in {"raw", "raw-activation"}:
        dictionary_id = f"raw-activation/{trace.behavior.model}"
    elif normalized not in {"custom", "neuronpedia"}:
        typer.secho(f"Unknown dictionary: {dictionary}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if source and source.lower() == "neuronpedia":
        dictionary_id = f"{dictionary_id}+neuronpedia"

    try:
        feature_artifact = build_raw_activation_features(
            run_id=f"{store.run_id}-features",
            backend=trace.backend,
            behavior=trace.behavior,
            activation_path=store.path("activations.npz"),
            dictionary_id=dictionary_id,
            source=source_name,
            top_k=top_k,
            layer=layer,
            stream=stream,
        )
    except Exception as exc:
        typer.secho(f"features failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    artifact_refs = {
        "features": "features.json",
        "evidence": "feature_evidence.jsonl",
        "report_md": "features.md",
        "report_html": "features.html",
    }
    feature_artifact = feature_artifact.model_copy(update={"artifact_refs": artifact_refs})
    store.write_json("features.json", feature_artifact)
    evidence_lines = "\n".join(
        evidence.model_dump_json() for evidence in feature_artifact.evidence
    )
    store.write_text("feature_evidence.jsonl", evidence_lines + ("\n" if evidence_lines else ""))
    store.write_text("features.md", render_features_markdown(feature_artifact))
    store.write_text("features.html", render_html("Feature Report", render_features_markdown(feature_artifact)))
    typer.echo(f"Wrote feature artifacts to {store.root}")


@app.command("graph")
def graph_command(
    run_path: Annotated[Path, typer.Argument(help="Run directory containing trace.json.")],
    method: Annotated[
        str,
        typer.Option("--method", help="Graph method: meir."),
    ] = "meir",
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Graph backend: meir or circuit-tracer."),
    ] = None,
    import_path: Annotated[
        Path | None,
        typer.Option("--import", help="Import path for circuit-tracer JSON graphs."),
    ] = None,
    prune_threshold: Annotated[
        float,
        typer.Option("--prune-threshold", help="Drop edges/nodes below this absolute effect."),
    ] = 0.0,
) -> None:
    store, trace = _load_trace(run_path)
    selected = (backend or method).lower().replace("_", "-")
    if selected == "circuit-tracer":
        if import_path is None:
            typer.secho(
                "circuit-tracer backend requires --import path/to/graph.json.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        graph = import_circuit_tracer_graph(
            run_id=f"{store.run_id}-circuit-tracer-graph",
            path=import_path,
            behavior=trace.behavior,
        )
    elif selected != "meir":
        typer.secho(f"Unknown graph method/backend: {selected}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    else:
        localization = (
            store.read_model("localization.json", LocalizationArtifact)
            if store.path("localization.json").exists()
            else None
        )
        features = (
            store.read_model("features.json", FeatureArtifact)
            if store.path("features.json").exists()
            else None
        )
        graph = build_meir_graph(
            run_id=f"{store.run_id}-graph",
            trace=trace,
            localization=localization,
            features=features,
            prune_threshold=prune_threshold,
        )
    artifact_refs = {
        "graph": "graph.json",
        "graphml": "graph.graphml",
        "report_md": "graph.md",
        "report_html": "graph.html",
    }
    graph = graph.model_copy(update={"artifact_refs": artifact_refs})
    store.write_json("graph.json", graph)
    store.write_text("graph.graphml", graph_to_graphml(graph))
    store.write_text("graph.md", render_graph_markdown(graph))
    store.write_text("graph.html", render_html("Evidence Graph Report", render_graph_markdown(graph)))
    typer.echo(f"Wrote graph artifacts to {store.root}")


@app.command("diff")
def diff_command(
    run_a: Annotated[
        Path | None,
        typer.Option("--run-a", help="Existing run directory for model/checkpoint A."),
    ] = None,
    run_b: Annotated[
        Path | None,
        typer.Option("--run-b", help="Existing run directory for model/checkpoint B."),
    ] = None,
    model_a: Annotated[str | None, typer.Option("--model-a", help="Model A label.")] = None,
    model_b: Annotated[str | None, typer.Option("--model-b", help="Model B label.")] = None,
    suite: Annotated[Path | None, typer.Option("--suite", help="Prompt suite path for metadata.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output diff directory.")] = Path("runs/diff"),
) -> None:
    if run_a is None or run_b is None:
        if model_a and model_b:
            typer.secho(
                "Direct model diff execution is planned after backend batch tracing; pass --run-a and --run-b for v0.8.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        typer.secho("diff requires --run-a and --run-b in v0.8.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    _, trace_a = _load_trace(run_a)
    _, trace_b = _load_trace(run_b)
    store = ArtifactStore.create(out)
    diff = diff_traces(store.run_id, trace_a, trace_b)
    if suite is not None:
        diff.warnings.append(f"Suite metadata recorded: {suite}")
    artifact_refs = {"diff": "diff.json", "report_md": "diff.md", "report_html": "diff.html"}
    diff = diff.model_copy(update={"artifact_refs": artifact_refs})
    store.write_json("diff.json", diff)
    store.write_text("diff.md", render_diff_markdown(diff))
    store.write_text("diff.html", render_html("Model Diff Report", render_diff_markdown(diff)))
    typer.echo(f"Wrote diff artifacts to {store.root}")


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
