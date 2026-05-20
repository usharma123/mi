from __future__ import annotations

import itertools
import json
from pathlib import Path

import yaml

from mi.core.schema import PromptVariant


def load_prompt_family(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prompt family must be a YAML object.")
    return payload


def generate_variants(family: dict) -> list[PromptVariant]:
    family_name = str(family.get("family", "family"))
    templates = family.get("templates") or []
    variables = family.get("variables") or {}
    target_template = family.get("target_template")
    if not templates:
        raise ValueError("Prompt family must define at least one template.")
    if not isinstance(variables, dict):
        raise ValueError("Prompt family variables must be an object.")
    keys = list(variables)
    values = [variables[key] for key in keys]
    variants: list[PromptVariant] = []
    for template_index, template in enumerate(templates):
        for combo_index, combo in enumerate(itertools.product(*values) if values else [()]):
            data = dict(zip(keys, combo))
            prompt = str(template).format(**data)
            target_text = str(target_template).format(**data) if target_template else None
            variants.append(
                PromptVariant(
                    id=f"{family_name}-{template_index}-{combo_index}",
                    prompt=prompt,
                    target_text=target_text,
                    metadata={"family": family_name, **data},
                )
            )
    return variants


def write_variants_jsonl(path: Path, variants: list[PromptVariant]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(variant.model_dump_json() + "\n" for variant in variants),
        encoding="utf-8",
    )


def load_variants_jsonl(path: Path) -> list[PromptVariant]:
    variants = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            variants.append(PromptVariant.model_validate(json.loads(line)))
    return variants
