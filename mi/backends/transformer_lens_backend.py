from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mi.backends.base import BackendUnavailable
from mi.core.schema import (
    ActivationSummary,
    BehaviorSpec,
    DirectLogitAttributionEntry,
    LogitLensEntry,
    TargetMetrics,
    TopPrediction,
    TraceArtifact,
)
from mi.methods.activation_capture import (
    activation_artifact_key,
    activation_ref_from_hook,
    should_capture_activation,
)


class TransformerLensBackend:
    name = "transformer-lens"

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

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
