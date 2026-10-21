# src/processing/translator.py
"""Script conversion for Serbian (Cyrillic ↔ Latin).

The retrieval corpus, the prompts and the chat UI all use Serbian Latin
script.  User questions and retrieved chunks can still be Cyrillic, and a
local model occasionally slips into Cyrillic on its own, so every text that
crosses the LLM boundary is converted here instead of trusted to the model.
"""

import re

import cyrtranslit
from langdetect import detect


_LATINIC_ALIASES = {"latinic", "latin", "latinc", "latinica"}
_CYRILLIC_ALIASES = {"cyrillic", "cyrilic", "cirilica", "ćirilica"}


def normalize_script(script: str | None, default: str = "latinic") -> str:
    """Map the spelling variants used by the API/settings to a script name."""
    value = (script or "").strip().lower()
    if value in _CYRILLIC_ALIASES:
        return "cyrillic"
    if value in _LATINIC_ALIASES:
        return "latinic"
    return default


class ScriptTranslator:
    def __init__(self):
        # All Cyrillic ranges: Serbian đ, ć, č, ž, š, lj, nj, dž are outside
        # the Russian а-я block, so a narrower pattern would miss them.
        self.cyrillic_pattern = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")

    def detect_script(self, text: str) -> str:
        """Detect if text contains Cyrillic."""
        if self.cyrillic_pattern.search(text or ""):
            return "cyrillic"
        return "latinic"

    def contains_cyrillic(self, text: str) -> bool:
        return self.detect_script(text) == "cyrillic"

    def cyrillic_to_latinic(self, text: str, lang: str = "sr") -> str:
        """
        Convert Cyrillic to Latinic.
        Supports: 'sr' (Serbian), 'mk' (Macedonian), 'bg' (Bulgarian)
        """
        if not text or self.detect_script(text) == "latinic":
            return text

        try:
            if lang == "sr":
                return cyrtranslit.to_latin(text, "sr")
            elif lang == "mk":
                return cyrtranslit.to_latin(text, "mk")
            else:
                # Auto-detect
                detected_lang = detect(text)
                if detected_lang in ["sr", "mk", "bg"]:
                    return cyrtranslit.to_latin(text, detected_lang)
                return text
        except Exception:
            # Fallback to Serbian
            return cyrtranslit.to_latin(text, "sr")

    def to_latinic(self, text: str, lang: str = "sr") -> str:
        """Alias used by the answer normaliser and the QA chain."""
        return self.cyrillic_to_latinic(text, lang)

    def latinic_to_cyrillic(self, text: str, lang: str = "sr") -> str:
        """Convert Latinic to Cyrillic (if needed)."""
        try:
            if not text:
                return text
            return cyrtranslit.to_cyrillic(text, "sr" if lang == "sr" else "mk")
        except Exception:
            return text

    def to_script(self, text: str, script: str = "latinic") -> str:
        """
        Return ``text`` in the requested script.

        Everything is normalised through Latin first: transliteration from
        Latin is lossless for Serbian, while converting Cyrillic → Latin →
        Cyrillic can change mixed-script output in unpredictable ways.
        """
        if not text:
            return text
        latin = self.to_latinic(text)
        if normalize_script(script) == "cyrillic":
            return self.latinic_to_cyrillic(latin)
        return latin

    def normalize_query(self, text: str) -> str:
        """Normalize query by converting to Latinic."""
        return self.cyrillic_to_latinic(text)
