# mi

`mi` is a library-first CLI for mechanistic interpretability traces. The v0.1 release focuses on a boring, reproducible foundation: run a local TransformerLens-compatible model, capture activations and next-token metrics, normalize the result into MEIR artifacts, and write Markdown/JSON reports.

```bash
mi trace \
  --model gpt2-small \
  --prompt "The capital of France is" \
  --target " Paris" \
  --out runs/france
```

This writes:

```text
runs/france/
  manifest.json
  trace.json
  tokens.json
  metrics.json
  activations.npz
  report.md
```

## v0.1 philosophy

`mi` treats natural-language labels as metadata, not proof. The first release reports token metrics, activation inventories, logit-lens summaries, and direct attribution estimates. Validated causal claims, prompt-family checks, SAE feature intervention, and graph imports are planned for later milestones.

## Commands

```bash
mi trace --model gpt2-small --prompt "The capital of France is" --target " Paris"
mi inspect runs/france --view logits
mi inspect runs/france --view logit-lens
mi inspect runs/france --view activations
mi localize runs/france --methods zero-ablation --top-k 20
mi localize runs/france \
  --corrupt-prompt "The capital of Germany is" \
  --methods zero-ablation,clean-to-corrupt-patch
mi report runs/france --format md,json
```

`mi localize` writes:

```text
runs/france/
  localization.json
  candidates.json
  evidence.jsonl
  localize.md
```

## Backends

The implemented v0.1/v0.2 path is TransformerLens because it exposes internal activations and hook-based interventions. Ollama-style generation APIs are not enough for causal tracing by themselves: `mi` needs residual streams, component outputs, and intervention hooks. Models served by Ollama can become useful when their underlying weights are loaded through a PyTorch/Hugging Face/NNsight-style backend that exposes those tensors.

## Development

```bash
uv sync --group dev
uv run pytest
```

Integration tests that load model libraries or weights are marked separately:

```bash
uv run pytest -m integration
```
