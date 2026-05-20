"""Feature dictionary adapters.

v0.1 ships the interfaces and explicit stubs; causal SAE feature workflows arrive
after the local trace engine is stable.
"""

from mi.dictionaries.base import FeatureDictionary, FeatureLookupResult

__all__ = ["FeatureDictionary", "FeatureLookupResult"]
