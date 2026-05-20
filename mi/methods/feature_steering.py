from __future__ import annotations


class FeatureSteeringNotImplemented(NotImplementedError):
    pass


def require_feature_steering() -> None:
    raise FeatureSteeringNotImplemented("Feature steering is planned for mi v0.3.")
