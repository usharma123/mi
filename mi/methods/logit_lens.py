from __future__ import annotations

from mi.core.schema import LogitLensEntry


def sort_logit_lens(entries: list[LogitLensEntry]) -> list[LogitLensEntry]:
    return sorted(entries, key=lambda item: (item.layer, item.stream))
