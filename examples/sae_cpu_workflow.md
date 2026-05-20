# CPU SAE Workflow

This is the reproducible `gpt2-small` path for the lab-grade SAE milestone.

```bash
mi trace \
  --model gpt2-small \
  --prompt "The capital city of France is called" \
  --target " Paris" \
  --out runs/france \
  --device cpu

mi features runs/france \
  --dictionary saelens \
  --sae-release gpt2-small-res-jb \
  --sae-id blocks.8.hook_resid_pre \
  --top-k 10 \
  --feature-ablation \
  --feature-steering \
  --steer-scale 2.0 \
  --device cpu

mi fuzz examples/families/capital_cities.yml --out runs/france/variants.jsonl

mi validate runs/france \
  --claims examples/claims/france_paris.yml \
  --variants runs/france/variants.jsonl \
  --min-variant-pass-rate 0.7 \
  --controls random,same-layer,wrong-target \
  --seed 0 \
  --device cpu

mi test "examples/claims/*.yml" \
  --model gpt2-small \
  --device cpu \
  --out runs/regression
```

Neuronpedia labels can be added with `--metadata neuronpedia`; they are cached metadata and do not count as evidence.
