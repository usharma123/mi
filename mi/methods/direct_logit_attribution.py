from __future__ import annotations

from mi.core.schema import DirectLogitAttributionEntry


def top_direct_attributions(
    entries: list[DirectLogitAttributionEntry], limit: int = 20
) -> list[DirectLogitAttributionEntry]:
    return sorted(entries, key=lambda item: abs(item.contribution), reverse=True)[:limit]
