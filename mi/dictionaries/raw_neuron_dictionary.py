from __future__ import annotations

from mi.dictionaries.base import NotImplementedFeatureDictionary


class RawNeuronDictionary(NotImplementedFeatureDictionary):
    dictionary_id = "raw-neuron"

    def top_features(self, activation, *, top_k: int):
        raise NotImplementedError("Raw-neuron feature views are planned for a later release.")
