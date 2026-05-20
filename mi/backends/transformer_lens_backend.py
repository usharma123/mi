from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mi.backends.base import BackendUnavailable
from mi.core.schema import (
    ActivationSummary,
    BehaviorSpec,
    ControlSummary,
    DirectLogitAttributionEntry,
    Evidence,
    LocalizationArtifact,
    LocalizationCandidate,
    LogitLensEntry,
    TargetMetrics,
    TopPrediction,
    TraceArtifact,
)
from mi.core.schema import ActivationRef
from mi.methods.activation_capture import (
    activation_artifact_key,
    activation_ref_from_hook,
    should_capture_activation,
)
from mi.methods.localization import resolve_position


class TransformerLensBackend:
    name = "transformer-lens"
    LOCALIZE_HOOKS = {
        "resid_post": "blocks.{layer}.hook_resid_post",
        "attn_out": "blocks.{layer}.hook_attn_out",
        "mlp_out": "blocks.{layer}.hook_mlp_out",
    }

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def capabilities(self) -> dict[str, bool]:
        return {"trace": True, "localize": True, "features": True, "graph": True}

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise BackendUnavailable(
                "TransformerLens is required for --backend transformer-lens. "
                "Install project dependencies with `uv sync`."
            ) from exc

        device = self.device
        if device in {None, "auto"}:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        return HookedTransformer.from_pretrained(self.model_name, device=device)

    def trace(
        self,
        behavior: BehaviorSpec,
        activation_path: Path,
        *,
        run_id: str,
        top_k: int,
    ) -> TraceArtifact:
        import torch

        model = self.model
        tokens_tensor = model.to_tokens(behavior.prompt)
        token_ids = [int(token_id) for token_id in tokens_tensor[0].detach().cpu().tolist()]
        tokens = list(model.to_str_tokens(behavior.prompt))

        logits, cache = model.run_with_cache(tokens_tensor)
        last_logits = logits[0, -1].detach()
        probs = torch.softmax(last_logits.float(), dim=-1)

        warnings: list[str] = []
        target_id: int | None = None
        target_token: str | None = None
        if behavior.target_text:
            target_ids = self._target_token_ids(behavior.target_text)
            if not target_ids:
                warnings.append("Target text tokenized to zero tokens; target metrics were skipped.")
            else:
                target_id = target_ids[0]
                target_token = self._decode_token(target_id)
                if len(target_ids) > 1:
                    warnings.append(
                        "Target text tokenized to multiple tokens; v0.1 reports the first "
                        f"target token only: {target_token!r}."
                    )

        behavior = behavior.model_copy(update={"target_token": target_token})
        top_predictions = self._top_predictions(last_logits, probs, top_k)
        target_metrics = self._target_metrics(last_logits, probs, target_id, target_token)
        final_position = len(token_ids) - 1
        activation_inventory = self._write_activations(cache, activation_path, final_position)
        logit_lens = self._logit_lens(cache, target_id, warnings)
        direct_logit_attribution = self._direct_logit_attribution(cache, target_id, final_position)

        return TraceArtifact(
            id=run_id,
            backend=self.name,
            behavior=behavior,
            token_ids=token_ids,
            tokens=tokens,
            target=target_metrics,
            top_predictions=top_predictions,
            logit_lens=logit_lens,
            direct_logit_attribution=direct_logit_attribution,
            activation_inventory=activation_inventory,
            warnings=warnings,
        )

    def localize(
        self,
        behavior: BehaviorSpec,
        *,
        run_id: str,
        corrupt_prompt: str | None,
        methods: set[str],
        streams: set[str],
        controls: set[str],
        position: str,
        seed: int,
        top_k: int,
    ) -> LocalizationArtifact:
        import torch

        model = self.model
        rng = np.random.default_rng(seed)
        warnings: list[str] = []
        target_id: int | None = None
        target_token: str | None = behavior.target_token
        if behavior.target_text:
            target_ids = self._target_token_ids(behavior.target_text)
            if target_ids:
                target_id = target_ids[0]
                target_token = self._decode_token(target_id)
                if len(target_ids) > 1:
                    warnings.append(
                        "Target text tokenized to multiple tokens; localization uses the first "
                        f"target token only: {target_token!r}."
                    )
            else:
                warnings.append("Target text tokenized to zero tokens; localization skipped.")

        behavior = behavior.model_copy(update={"target_token": target_token})
        if target_id is None:
            return LocalizationArtifact(
                id=run_id,
                backend=self.name,
                behavior=behavior,
                corrupt_prompt=corrupt_prompt,
                warnings=warnings,
            )

        clean_tokens = model.to_tokens(behavior.prompt)
        clean_token_ids = [int(token_id) for token_id in clean_tokens[0].detach().cpu().tolist()]
        clean_str_tokens = list(model.to_str_tokens(behavior.prompt))
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        clean_last_logits = clean_logits[0, -1].detach()
        clean_probs = torch.softmax(clean_last_logits.float(), dim=-1)
        clean_target = self._target_metrics(clean_last_logits, clean_probs, target_id, target_token)
        if clean_target is not None and (clean_target.rank > 10 or clean_target.probability < 0.05):
            warnings.append(
                "Target baseline is weak: "
                f"rank={clean_target.rank}, probability={clean_target.probability:.6f}."
            )

        corrupt_tokens = model.to_tokens(corrupt_prompt) if corrupt_prompt else None
        corrupt_token_ids: list[int] | None = None
        corrupt_str_tokens: list[str] | None = None
        corrupt_target = None
        corrupt_cache = None
        corrupt_baseline_logit = None
        corrupt_position = None
        corrupt_last_logits = None
        if corrupt_tokens is not None:
            corrupt_token_ids = [
                int(token_id) for token_id in corrupt_tokens[0].detach().cpu().tolist()
            ]
            corrupt_str_tokens = list(model.to_str_tokens(corrupt_prompt))
            corrupt_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)
            corrupt_last_logits = corrupt_logits[0, -1].detach()
            corrupt_probs = torch.softmax(corrupt_last_logits.float(), dim=-1)
            corrupt_target = self._target_metrics(
                corrupt_last_logits, corrupt_probs, target_id, target_token
            )
            corrupt_baseline_logit = float(corrupt_last_logits[target_id].item())
            if len(clean_token_ids) != len(corrupt_token_ids):
                warnings.append(
                    "Clean and corrupt prompts have different token lengths; position alignment may be ambiguous."
                )

        try:
            position_selection = resolve_position(
                position,
                clean_token_ids,
                clean_str_tokens,
                corrupt_token_ids=corrupt_token_ids,
                corrupt_tokens=corrupt_str_tokens,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid --position value: {exc}") from exc
        warnings.extend(position_selection.warnings)
        clean_position = position_selection.clean_position
        corrupt_position = position_selection.corrupt_position

        candidates: list[LocalizationCandidate] = []
        evidence: list[Evidence] = []
        control_effects_by_key: dict[tuple[str, str], dict[str, list[float]]] = {}
        selected_streams = streams or set(self.LOCALIZE_HOOKS)
        unknown_streams = selected_streams - set(self.LOCALIZE_HOOKS)
        if unknown_streams:
            warnings.append(f"Unsupported localization stream(s): {', '.join(sorted(unknown_streams))}.")
        selected_streams = selected_streams & set(self.LOCALIZE_HOOKS)
        baseline_logit = float(clean_last_logits[target_id].item())
        wrong_target_id = self._top_token_id_excluding(clean_last_logits, {target_id})
        wrong_corrupt_id = (
            self._top_token_id_excluding(corrupt_last_logits, {target_id})
            if corrupt_last_logits is not None
            else None
        )

        n_layers = int(getattr(model.cfg, "n_layers", 0))
        for layer in range(n_layers):
            for stream in sorted(selected_streams):
                hook_name = self.LOCALIZE_HOOKS[stream].format(layer=layer)
                if hook_name not in self._cache_dict(clean_cache):
                    continue
                target_ref = self._localization_target(layer, clean_position, stream)

                if "zero_ablation" in methods:
                    ablated_logits = model.run_with_hooks(
                        clean_tokens,
                        fwd_hooks=[(hook_name, self._zero_position_hook(clean_position))],
                    )
                    after_logit = float(ablated_logits[0, -1, target_id].detach().item())
                    effect = baseline_logit - after_logit
                    candidate = LocalizationCandidate(
                        method="zero_ablation",
                        target=target_ref,
                        hook_name=hook_name,
                        metric_before=baseline_logit,
                        metric_after=after_logit,
                        effect=float(effect),
                    )
                    candidates.append(candidate)
                    control_key = ("zero_ablation", hook_name)
                    control_effects_by_key.setdefault(control_key, {})
                    if "wrong-target" in controls and wrong_target_id is not None:
                        wrong_before = float(clean_last_logits[wrong_target_id].item())
                        wrong_after = float(
                            ablated_logits[0, -1, wrong_target_id].detach().item()
                        )
                        control_effects_by_key[control_key].setdefault(
                            "wrong-target", []
                        ).append(wrong_before - wrong_after)
                    if (
                        "wrong-corrupt" in controls
                        and corrupt_tokens is not None
                        and corrupt_baseline_logit is not None
                        and corrupt_position is not None
                    ):
                        corrupt_ablated_logits = model.run_with_hooks(
                            corrupt_tokens,
                            fwd_hooks=[
                                (hook_name, self._zero_position_hook(corrupt_position))
                            ],
                        )
                        control_effects_by_key[control_key].setdefault(
                            "wrong-corrupt", []
                        ).append(
                            corrupt_baseline_logit
                            - float(
                                corrupt_ablated_logits[0, -1, target_id].detach().item()
                            )
                        )
                    evidence.append(
                        Evidence(
                            id=f"ev_{len(evidence) + 1}",
                            method="zero_ablation",
                            target=target_ref,
                            metric_before=baseline_logit,
                            metric_after=after_logit,
                            delta=float(after_logit - baseline_logit),
                            artifact_refs=["localization.json"],
                        )
                    )

                if (
                    "clean_to_corrupt_patch" in methods
                    and corrupt_tokens is not None
                    and corrupt_cache is not None
                    and corrupt_baseline_logit is not None
                    and corrupt_position is not None
                ):
                    clean_value = self._cache_dict(clean_cache)[hook_name][
                        :, clean_position : clean_position + 1, :
                    ].detach()
                    patched_logits = model.run_with_hooks(
                        corrupt_tokens,
                        fwd_hooks=[
                            (
                                hook_name,
                                self._patch_position_hook(corrupt_position, clean_value),
                            )
                        ],
                    )
                    after_logit = float(patched_logits[0, -1, target_id].detach().item())
                    target_ref = self._localization_target(layer, corrupt_position, stream)
                    effect = after_logit - corrupt_baseline_logit
                    candidate = LocalizationCandidate(
                        method="clean_to_corrupt_patch",
                        target=target_ref,
                        hook_name=hook_name,
                        metric_before=corrupt_baseline_logit,
                        metric_after=after_logit,
                        effect=float(effect),
                    )
                    candidates.append(candidate)
                    control_key = ("clean_to_corrupt_patch", hook_name)
                    control_effects_by_key.setdefault(control_key, {})
                    if (
                        "wrong-target" in controls
                        and wrong_target_id is not None
                        and corrupt_last_logits is not None
                    ):
                        wrong_before = float(corrupt_last_logits[wrong_target_id].item())
                        wrong_after = float(
                            patched_logits[0, -1, wrong_target_id].detach().item()
                        )
                        control_effects_by_key[control_key].setdefault(
                            "wrong-target", []
                        ).append(wrong_after - wrong_before)
                    if (
                        "wrong-corrupt" in controls
                        and wrong_corrupt_id is not None
                        and corrupt_last_logits is not None
                    ):
                        wrong_before = float(corrupt_last_logits[wrong_corrupt_id].item())
                        wrong_after = float(
                            patched_logits[0, -1, wrong_corrupt_id].detach().item()
                        )
                        control_effects_by_key[control_key].setdefault(
                            "wrong-corrupt", []
                        ).append(wrong_after - wrong_before)
                    evidence.append(
                        Evidence(
                            id=f"ev_{len(evidence) + 1}",
                            method="clean_to_corrupt_patch",
                            target=target_ref,
                            metric_before=corrupt_baseline_logit,
                            metric_after=after_logit,
                            delta=float(after_logit - corrupt_baseline_logit),
                            artifact_refs=["localization.json"],
                        )
                    )

        if "clean_to_corrupt_patch" in methods and corrupt_prompt is None:
            warnings.append("clean_to_corrupt_patch requested without --corrupt-prompt; skipped.")
        if (
            "clean_to_corrupt_patch" in methods
            and corrupt_prompt is not None
            and corrupt_position is None
        ):
            warnings.append("clean_to_corrupt_patch requested but no corrupt position was resolved; skipped.")

        candidates = self._attach_control_summaries(
            candidates,
            controls=controls,
            control_effects_by_key=control_effects_by_key,
            rng=rng,
        )
        ranked_candidates: list[LocalizationCandidate] = []
        for method in ("zero_ablation", "clean_to_corrupt_patch"):
            method_candidates = [
                item for item in candidates if item.method == method
            ]
            method_candidates = sorted(
                method_candidates, key=lambda item: abs(item.effect), reverse=True
            )[:top_k]
            ranked_candidates.extend(
                item.model_copy(update={"rank": rank})
                for rank, item in enumerate(method_candidates, start=1)
            )
        return LocalizationArtifact(
            id=run_id,
            backend=self.name,
            behavior=behavior,
            corrupt_prompt=corrupt_prompt,
            target=clean_target,
            corrupt_target=corrupt_target,
            candidates=ranked_candidates,
            evidence=evidence,
            warnings=warnings,
        )

    def _attach_control_summaries(
        self,
        candidates: list[LocalizationCandidate],
        *,
        controls: set[str],
        control_effects_by_key: dict[tuple[str, str], dict[str, list[float]]],
        rng: np.random.Generator,
    ) -> list[LocalizationCandidate]:
        if not controls:
            return candidates

        updated: list[LocalizationCandidate] = []
        for candidate in candidates:
            control_effects = {
                name: list(values)
                for name, values in control_effects_by_key.get(
                    (candidate.method, candidate.hook_name), {}
                ).items()
            }
            if "same-layer" in controls:
                same_layer = [
                    other.effect
                    for other in candidates
                    if other.method == candidate.method
                    and other.target.layer == candidate.target.layer
                    and other.hook_name != candidate.hook_name
                ]
                if same_layer:
                    control_effects["same-layer"] = same_layer
            if "random" in controls:
                random_pool = [
                    other.effect
                    for other in candidates
                    if other.method == candidate.method and other.hook_name != candidate.hook_name
                ]
                if random_pool:
                    sample_size = min(5, len(random_pool))
                    indices = rng.choice(len(random_pool), size=sample_size, replace=False)
                    control_effects["random"] = [random_pool[int(index)] for index in indices]

            flat_effects = [
                abs(effect)
                for values in control_effects.values()
                for effect in values
            ]
            summary = ControlSummary(
                control_mean=float(np.mean(flat_effects)) if flat_effects else None,
                control_max=float(np.max(flat_effects)) if flat_effects else None,
                control_count=len(flat_effects),
                specificity_passed=(
                    abs(candidate.effect) > float(np.max(flat_effects))
                    if flat_effects
                    else None
                ),
                control_effects={
                    name: [float(effect) for effect in values]
                    for name, values in control_effects.items()
                },
            )
            updated.append(
                candidate.model_copy(
                    update={
                        "controls": sorted(controls),
                        "control_summary": summary,
                    }
                )
            )
        return updated

    def _target_token_ids(self, target_text: str) -> list[int]:
        target_tokens = self.model.to_tokens(target_text, prepend_bos=False)
        return [int(token_id) for token_id in target_tokens[0].detach().cpu().tolist()]

    def _decode_token(self, token_id: int) -> str:
        try:
            return str(self.model.to_string([token_id]))
        except Exception:
            return str(self.model.tokenizer.decode([token_id]))

    def _top_predictions(self, logits: Any, probs: Any, top_k: int) -> list[TopPrediction]:
        import torch

        limit = min(max(top_k, 1), int(logits.shape[-1]))
        values, indices = torch.topk(logits, k=limit)
        predictions: list[TopPrediction] = []
        for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            token_id = int(index)
            predictions.append(
                TopPrediction(
                    token_id=token_id,
                    token=self._decode_token(token_id),
                    logit=float(value),
                    probability=float(probs[token_id].item()),
                    rank=rank,
                )
            )
        return predictions

    def _target_metrics(
        self,
        logits: Any,
        probs: Any,
        target_id: int | None,
        target_token: str | None,
    ) -> TargetMetrics | None:
        if target_id is None or target_token is None:
            return None
        rank = int((logits > logits[target_id]).sum().item() + 1)
        return TargetMetrics(
            token_id=target_id,
            token=target_token,
            logit=float(logits[target_id].item()),
            probability=float(probs[target_id].item()),
            rank=rank,
        )

    def _top_token_id_excluding(self, logits: Any, excluded: set[int]) -> int | None:
        import torch

        if logits is None:
            return None
        sorted_ids = torch.argsort(logits, descending=True)
        for token_id in sorted_ids.tolist():
            token_id = int(token_id)
            if token_id not in excluded:
                return token_id
        return None

    def _localization_target(self, layer: int, position: int, stream: str) -> ActivationRef:
        return ActivationRef(layer=layer, position=position, stream=stream)

    def _zero_position_hook(self, position: int):
        def hook(value, hook):
            patched = value.clone()
            patched[:, position, :] = 0
            return patched

        return hook

    def _patch_position_hook(self, position: int, clean_value: Any):
        def hook(value, hook):
            patched = value.clone()
            patched[:, position : position + 1, :] = clean_value.to(
                device=value.device, dtype=value.dtype
            )
            return patched

        return hook

    def _cache_dict(self, cache: Any) -> dict[str, Any]:
        return getattr(cache, "cache_dict", cache)

    def _write_activations(
        self, cache: Any, activation_path: Path, final_position: int
    ) -> list[ActivationSummary]:
        arrays: dict[str, np.ndarray] = {}
        inventory: list[ActivationSummary] = []
        for name, tensor in self._cache_dict(cache).items():
            if not should_capture_activation(name):
                continue
            artifact_key = activation_artifact_key(name)
            cpu_tensor = tensor.detach().cpu()
            if str(cpu_tensor.dtype) in {"torch.bfloat16", "torch.float16"}:
                cpu_tensor = cpu_tensor.float()
            array = cpu_tensor.numpy()
            arrays[artifact_key] = array
            inventory.append(
                ActivationSummary(
                    name=name,
                    shape=[int(dim) for dim in array.shape],
                    dtype=str(array.dtype),
                    artifact_key=artifact_key,
                    ref=activation_ref_from_hook(name, final_position),
                )
            )

        activation_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(activation_path, **arrays)
        return inventory

    def _logit_lens(
        self, cache: Any, target_id: int | None, warnings: list[str]
    ) -> list[LogitLensEntry]:
        import torch

        entries: list[LogitLensEntry] = []
        n_layers = int(getattr(self.model.cfg, "n_layers", 0))
        for layer in range(n_layers):
            name = f"blocks.{layer}.hook_resid_post"
            if name not in self._cache_dict(cache):
                continue
            resid = self._cache_dict(cache)[name][:, -1:, :]
            try:
                lens_logits = self.model.unembed(self.model.ln_final(resid))[0, 0].detach()
            except Exception as exc:
                warnings.append(f"Logit lens failed at layer {layer}: {exc}")
                continue

            top_logit, top_token_id = torch.max(lens_logits, dim=-1)
            target_logit = None
            target_rank = None
            if target_id is not None:
                target_logit = float(lens_logits[target_id].item())
                target_rank = int((lens_logits > lens_logits[target_id]).sum().item() + 1)

            entries.append(
                LogitLensEntry(
                    layer=layer,
                    stream="resid_post",
                    target_logit=target_logit,
                    target_rank=target_rank,
                    top_token=self._decode_token(int(top_token_id.item())),
                    top_token_id=int(top_token_id.item()),
                    top_logit=float(top_logit.item()),
                )
            )

        final_logits = self.model.unembed(
            self.model.ln_final(self._cache_dict(cache)[f"blocks.{n_layers - 1}.hook_resid_post"][:, -1:, :])
        )[0, 0].detach() if n_layers > 0 and f"blocks.{n_layers - 1}.hook_resid_post" in self._cache_dict(cache) else None
        if final_logits is not None:
            top_logit, top_token_id = torch.max(final_logits, dim=-1)
            entries.append(
                LogitLensEntry(
                    layer=n_layers,
                    stream="final",
                    target_logit=float(final_logits[target_id].item()) if target_id is not None else None,
                    target_rank=int((final_logits > final_logits[target_id]).sum().item() + 1)
                    if target_id is not None
                    else None,
                    top_token=self._decode_token(int(top_token_id.item())),
                    top_token_id=int(top_token_id.item()),
                    top_logit=float(top_logit.item()),
                )
            )
        return entries

    def _direct_logit_attribution(
        self, cache: Any, target_id: int | None, final_position: int
    ) -> list[DirectLogitAttributionEntry]:
        if target_id is None or not hasattr(self.model, "W_U"):
            return []

        import torch

        target_direction = self.model.W_U[:, target_id].detach().float()
        entries: list[DirectLogitAttributionEntry] = []
        n_layers = int(getattr(self.model.cfg, "n_layers", 0))
        cache_dict = self._cache_dict(cache)
        for layer in range(n_layers):
            for component, name in (
                ("attn_out", f"blocks.{layer}.hook_attn_out"),
                ("mlp_out", f"blocks.{layer}.hook_mlp_out"),
            ):
                if name not in cache_dict:
                    continue
                vector = cache_dict[name][0, final_position].detach().float()
                contribution = torch.dot(vector, target_direction).item()
                entries.append(
                    DirectLogitAttributionEntry(
                        layer=layer,
                        component=component,
                        position=final_position,
                        contribution=float(contribution),
                    )
                )
        return entries
