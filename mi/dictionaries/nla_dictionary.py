from __future__ import annotations

from mi.dictionaries.base import NotImplementedFeatureDictionary


class NLADictionary(NotImplementedFeatureDictionary):
    dictionary_id = "nla"

    def top_features(self, activation, *, top_k: int):
        raise NotImplementedError(
            "Natural-language autoencoder labels are planned for a later release and will be weak metadata."
        )
