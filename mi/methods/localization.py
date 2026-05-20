from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SUPPORTED_CONTROLS = {"random", "same-layer", "wrong-target", "wrong-corrupt"}


@dataclass(frozen=True)
class PositionSelection:
    clean_position: int
    corrupt_position: int | None
    warnings: tuple[str, ...] = ()


def parse_controls(value: str | None) -> set[str]:
    if not value:
        return set()
    controls = {item.strip().lower() for item in value.split(",") if item.strip()}
    controls = {item.replace("_", "-") for item in controls}
    unknown = controls - SUPPORTED_CONTROLS
    if unknown:
        raise ValueError(f"Unknown control(s): {', '.join(sorted(unknown))}")
    return controls


def resolve_position(
    spec: str,
    clean_token_ids: list[int],
    clean_tokens: list[str],
    *,
    corrupt_token_ids: list[int] | None = None,
    corrupt_tokens: list[str] | None = None,
) -> PositionSelection:
    normalized = spec.strip()
    if not normalized or normalized == "final":
        return PositionSelection(
            clean_position=len(clean_token_ids) - 1,
            corrupt_position=len(corrupt_token_ids) - 1 if corrupt_token_ids else None,
        )

    if normalized == "first-diff":
        if not corrupt_token_ids:
            return PositionSelection(
                clean_position=len(clean_token_ids) - 1,
                corrupt_position=None,
                warnings=("first-diff requested without a corrupt prompt; using final clean position.",),
            )
        for index, (clean_id, corrupt_id) in enumerate(zip(clean_token_ids, corrupt_token_ids)):
            if clean_id != corrupt_id:
                return PositionSelection(clean_position=index, corrupt_position=index)
        fallback = min(len(clean_token_ids), len(corrupt_token_ids)) - 1
        return PositionSelection(
            clean_position=fallback,
            corrupt_position=fallback,
            warnings=("No differing token found for first-diff; using final aligned position.",),
        )

    if normalized.startswith("index:"):
        raw_index = normalized.removeprefix("index:")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise ValueError(f"Invalid position index: {raw_index}") from exc
        if index < 0 or index >= len(clean_token_ids):
            raise ValueError(f"Clean position index out of range: {index}")
        corrupt_position = None
        if corrupt_token_ids is not None:
            if index >= len(corrupt_token_ids):
                raise ValueError(f"Corrupt position index out of range: {index}")
            corrupt_position = index
        return PositionSelection(clean_position=index, corrupt_position=corrupt_position)

    if normalized.startswith("token:"):
        token_text = normalized.removeprefix("token:")
        clean_matches = _token_matches(clean_tokens, token_text)
        if not clean_matches:
            raise ValueError(f"Token position not found in clean prompt: {token_text!r}")
        warnings: list[str] = []
        if len(clean_matches) > 1:
            warnings.append(
                f"Token position {token_text!r} matched multiple clean tokens; using first match."
            )
        corrupt_position = None
        if corrupt_tokens is not None:
            corrupt_matches = _token_matches(corrupt_tokens, token_text)
            if corrupt_matches:
                if len(corrupt_matches) > 1:
                    warnings.append(
                        f"Token position {token_text!r} matched multiple corrupt tokens; using first match."
                    )
                corrupt_position = corrupt_matches[0]
            else:
                warnings.append(
                    f"Token position {token_text!r} was not found in corrupt prompt; using clean position for patch source only."
                )
        return PositionSelection(
            clean_position=clean_matches[0],
            corrupt_position=corrupt_position,
            warnings=tuple(warnings),
        )

    raise ValueError(
        "Unsupported position spec. Use final, first-diff, index:N, or token:TEXT."
    )


def _token_matches(tokens: list[str], token_text: str) -> list[int]:
    stripped = token_text.strip()
    return [
        index
        for index, token in enumerate(tokens)
        if token == token_text or token.strip() == stripped
    ]
