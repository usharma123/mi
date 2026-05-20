from __future__ import annotations

from mi.core.schema import Claim, Evidence


def evidence_delta_sum(claim: Claim, evidence: list[Evidence]) -> float:
    evidence_by_id = {item.id: item for item in evidence}
    return sum(evidence_by_id[item_id].delta for item_id in claim.evidence_ids if item_id in evidence_by_id)
