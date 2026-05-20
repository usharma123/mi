from __future__ import annotations

from mi.dictionaries.base import NotImplementedFeatureDictionary


class TranscoderDictionary(NotImplementedFeatureDictionary):
    dictionary_id = "transcoder"

    def top_features(self, activation, *, top_k: int):
        raise NotImplementedError("Transcoder feature mapping is planned for mi v0.6.")
