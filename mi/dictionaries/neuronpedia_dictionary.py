from __future__ import annotations

from mi.dictionaries.base import NotImplementedFeatureDictionary


class NeuronpediaDictionary(NotImplementedFeatureDictionary):
    dictionary_id = "neuronpedia"

    def top_features(self, activation, *, top_k: int):
        raise NotImplementedError(
            "Neuronpedia lookup is planned for mi v0.3 and will be treated as metadata, not evidence."
        )
