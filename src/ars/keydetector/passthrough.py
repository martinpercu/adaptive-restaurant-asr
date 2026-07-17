"""Pass-through keydetector for phase 1 (the real rule engine arrives in phase 5).

Honors the corrector seam: `correct(text, lang) -> (text, corrections)`. Returns no
corrections, so `final_text == raw_text` until confusion rules exist.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ars.contracts import Correction


@runtime_checkable
class Corrector(Protocol):
    def correct(self, text: str, lang: str) -> tuple[str, list[Correction]]: ...


class PassthroughKeydetector:
    def correct(self, text: str, lang: str) -> tuple[str, list[Correction]]:
        return text, []
