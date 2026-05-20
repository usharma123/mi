from __future__ import annotations

from mi.dictionaries.base import NotImplementedFeatureDictionary


class SAELensDictionary(NotImplementedFeatureDictionary):
    dictionary_id = "saelens"

    def top_features(self, activation, *, top_k: int):
        raise NotImplementedError("SAELens feature mapping is planned for mi v0.3.")
