"""Keydetector — AXIS 3: menu lexicon, phonetic keys, confusion-rule engine (phase 5).

Phase 1 uses `phonetics` for KER; the correction engine arrives in phase 5. The
pipeline wires a pass-through corrector until then.
"""

from ars.keydetector.phonetics import en_key, es_key, phonetic_key

__all__ = ["phonetic_key", "es_key", "en_key"]
