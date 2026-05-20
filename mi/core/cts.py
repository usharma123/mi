from __future__ import annotations

from mi.core.schema import ScoreArtifact, ValidationArtifact


def score_validation(validation: ValidationArtifact) -> ScoreArtifact:
    scores = {}
    for result in validation.results:
        effects = [test.effect for test in result.tests if test.effect is not None]
        passed = [test.passed for test in result.tests]
        attribution = min(max(sum(effects) / max(len(effects), 1), 0.0) / 10.0, 1.0) if effects else None
        robustness = (
            result.variant_pass_rate
            if result.variant_pass_rate is not None
            else (sum(1 for item in passed if item) / len(passed) if passed else None)
        )
        specificity_values = [
            1.0
            for test in result.tests
            if test.control_max is not None
            and (test.max_control_effect is None or test.control_max <= test.max_control_effect)
        ]
        specificity = (
            sum(specificity_values) / len(result.tests)
            if result.tests
            else None
        )
        available = [
            value
            for value in {
                "association": None,
                "attribution": attribution,
                "necessity": robustness,
                "sufficiency": robustness,
                "robustness": robustness,
                "specificity": specificity,
                "dictionary": None,
                "label_consistency": None,
            }.values()
            if value is not None
        ]
        cts = sum(available) / len(available) if available else None
        scores[result.claim_id] = {
            "association": None,
            "attribution": attribution,
            "necessity": robustness,
            "sufficiency": robustness,
            "robustness": robustness,
            "specificity": specificity,
            "dictionary": None,
            "label_consistency": None,
            "cts": cts,
        }
    return ScoreArtifact(id=f"{validation.id}-scores", scores=scores)
