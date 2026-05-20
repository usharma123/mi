from __future__ import annotations

import re

from mi.core.schema import ActivationRef

CAPTURE_HOOK_PATTERNS = (
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "hook_attn_out",
    "hook_mlp_out",
    "hook_pattern",
)

STREAM_BY_HOOK = {
    "hook_resid_pre": "resid_pre",
    "hook_resid_mid": "resid_mid",
    "hook_resid_post": "resid_post",
    "hook_attn_out": "attn_out",
    "hook_mlp_out": "mlp_out",
}


def should_capture_activation(name: str) -> bool:
    return any(pattern in name for pattern in CAPTURE_HOOK_PATTERNS)


def activation_artifact_key(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "__", name).strip("_")


def activation_ref_from_hook(name: str, position: int) -> ActivationRef | None:
    layer_match = re.search(r"blocks\.(\d+)\.", name)
    if not layer_match:
        return None

    stream = None
    for hook_name, candidate_stream in STREAM_BY_HOOK.items():
        if hook_name in name:
            stream = candidate_stream
            break

    if stream is None:
        return None

    return ActivationRef(layer=int(layer_match.group(1)), position=position, stream=stream)
