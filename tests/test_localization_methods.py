from __future__ import annotations

import pytest

from mi.methods.localization import parse_controls, resolve_position


def test_parse_controls_is_deterministic_and_rejects_unknown() -> None:
    assert parse_controls("random,same_layer,wrong-target") == {
        "random",
        "same-layer",
        "wrong-target",
    }
    with pytest.raises(ValueError, match="Unknown control"):
        parse_controls("random,unknown")


def test_resolve_position_variants() -> None:
    clean_ids = [1, 2, 3]
    corrupt_ids = [1, 4, 3]
    clean_tokens = ["The", " clean", " token"]
    corrupt_tokens = ["The", " corrupt", " token"]

    assert resolve_position("final", clean_ids, clean_tokens).clean_position == 2
    first_diff = resolve_position(
        "first-diff",
        clean_ids,
        clean_tokens,
        corrupt_token_ids=corrupt_ids,
        corrupt_tokens=corrupt_tokens,
    )
    assert first_diff.clean_position == 1
    assert first_diff.corrupt_position == 1
    assert resolve_position("index:1", clean_ids, clean_tokens).clean_position == 1
    assert resolve_position("token:clean", clean_ids, clean_tokens).clean_position == 1


def test_resolve_position_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        resolve_position("index:99", [1], [" only"])
