# src/processing/serbian_text.py
"""
Deterministic Serbian clean-up for LLM answers.

The local Qwen model writes good Serbian most of the time, but it can:

  * switch to Cyrillic (especially when the question or a retrieved chunk
    is Cyrillic), and
  * drop diacritics or use the wrong word form (``Racunarske`` instead of
    ``Računarske``, ``informativne tehnologije`` instead of the official
    ``informacione tehnologije``).

Script is fixed deterministically by transliteration.  Missing diacritics
are restored only from the retrieved context: when a word appears there in
exactly one accented form, that form wins.  Ambiguous words (both ``sto``
and ``što`` in the context) are left alone, so the corrector never guesses.
A short list of observed terminology mistakes is corrected explicitly.

``SerbianAnswerNormalizer`` also works token by token for SSE streaming: it
holds back only the last, possibly incomplete word, normalises complete
words, and flushes the rest at the end.  This keeps cyrillic multi-letter
mappings and consistent words from being split across chunks.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from .translator import normalize_script


_DIACRITICS = set("čćšžđČĆŠŽĐ")

# Cyrillic lookalikes that can survive Serbian transliteration (usually
# Ukrainian/Russian letters inside an otherwise Latin word) and Latin
# letters carrying accents that Serbian does not use.  Both are symptoms of
# a bad model token, not intentional text.
_CYRILLIC_LOOKALIKES = {
    "а": "a", "е": "e", "о": "o", "р": "r", "с": "c", "у": "u", "х": "h",
    "і": "i", "ј": "j", "ѕ": "s", "э": "e", "ё": "e", "ы": "y", "й": "j",
    "ї": "i", "є": "je", "ґ": "g", "џ": "dž", "љ": "lj", "њ": "nj",
    "А": "A", "Е": "E", "О": "O", "Р": "R", "С": "C", "У": "U", "Х": "H",
    "І": "I", "Ј": "J", "Ѕ": "S", "Э": "E", "Ё": "E", "Ы": "Y", "Й": "J",
    "Ї": "I", "Є": "Je", "Ґ": "G", "Џ": "Dž", "Љ": "Lj", "Њ": "Nj",
}
_SERBIAN_LATIN = set("čćšžđČĆŠŽĐ")

# Map a word to a diacritic-free lookup key.  ``đ`` folds to ``dj`` so an
# old-style answer ("djaci") can be matched to the context form ("đaci").
_FOLD = str.maketrans({
    "č": "c", "ć": "c", "š": "s", "ž": "z", "đ": "dj",
})

# A word is a run of Unicode letters, optionally joined by an apostrophe or
# hyphen.  Numbers and punctuation stay outside.
_WORD_RE = re.compile(r"[^\W\d_]+(?:[’'\-][^\W\d_]+)*", re.UNICODE)
_TRAILING_WORD_RE = re.compile(r"[^\W\d_]+$", re.UNICODE)
# URLs/e-mail addresses must never be transliterated or "corrected".
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+",
    re.IGNORECASE,
)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_BARE_CITATION_RE = re.compile(r"(?<!!)\[(\d{1,2})\](?!\()")
_CITATION_CASE_RE = re.compile(r"\[izvor\s*(\d{1,2})\]", re.IGNORECASE)
_DOUBLE_CITATION_RE = re.compile(
    r"\[\[\s*(?:izvor\s*)?(\d{1,2})\s*\]\]", re.IGNORECASE,
)
_MAX_CITATION_HOLD = 24  # characters to wait for a closing "]"

# Boilerplate that only says where the answer came from.  It carries no
# information and the user asked for direct answers, so it is removed and
# the following word is capitalised.  The patterns accept missing
# diacritics ("prilozenog") because the model sometimes drops them.
_CONTEXT_INTRO_RE = re.compile(
    r"(?i)(^|(?<=[.!?]\s))"
    r"(?:(?:na\s+osnovu|prema|iz|u)\s+)?"
    r"(?:(?:prilo[žz]en\w*|dostavljen\w*|dostupn\w*|naveden\w*|dat\w*"
    r"|ponu[đd]en\w*)\s+)?"
    r"(?:kontekst\w*|dokumentacij\w*|izvor\w*|informacij\w*|podac\w*)"
    r"\s*[,:;—–-]\s*"
    r"([a-zčćšžđ])"
)
_CONTEXT_PARENTHETICAL_RE = re.compile(
    r"(?i)\s*,\s*(?:prema|na\s+osnovu|u)\s+"
    r"(?:prilo[žz]en\w*\s+)?(?:kontekst\w*|dokumentacij\w*|izvor\w*)\s*,"
)
_CONTEXT_CLAIM_RE = re.compile(
    r"(?i)(^|(?<=[.!?]\s))"
    r"(?:u\s+)?(?:prilo[žz]en\w*\s+)?(?:kontekst\w*|dokumentacij\w*)\s+"
    r"(?:se\s+)?(?:navodi|kaže|pokazuje|potvrđuje|piše)\s+da\s+"
    r"(?!je\b|su\b|se\b)([a-zčćšžđ])"
)
_CONTEXT_NO_INFO_RE = re.compile(
    r"(?is)^\s*(?:u\s+)?(?:prilo[žz]en\w*\s+)?(?:kontekst\w*|dokumentacij\w*)\s+"
    r"(?:ne\s+(?:sadrži|navodi|pominje|ima)|nema)\b"
)
_MAX_CONTEXT_ONLY_ANSWER = 300  # chars; longer answers may carry real content

# Openings that may turn into a context-reference intro.  In streaming the
# first tokens are held until enough of the opening is known to decide
# whether it should be stripped; otherwise "Na osnovu priloženog konteksta"
# would already be on screen before the normaliser sees the whole phrase.
_OPENING_PHRASES = (
    "na osnovu", "prema", "iz prilo", "iz kontek", "iz dokumentacij",
    "iz izvor", "iz dostupn", "iz dostavljen", "u prilo", "u kontek",
    "u dokumentacij", "kontekst", "dokumentacij", "priložen", "prilozen",
)
_OPENING_HOLD_MAX = 120  # chars; never hold the first visible token longer

_INVISIBLE = str.maketrans({
    "\u00a0": " ",   # non-breaking space
    "\u2007": " ",   # figure space
    "\u202f": " ",   # narrow no-break space
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM
})


def fold_word(word: str) -> str:
    """Lowercase a word and remove Serbian diacritics for lookup."""
    return word.lower().translate(_FOLD)


def _strip_latin_accent(char: str) -> str:
    """``é``/``ė``/``í`` → ``e``/``e``/``i``; Serbian letters are kept."""
    if char in _SERBIAN_LATIN:
        return char
    decomposed = unicodedata.normalize("NFD", char)
    if decomposed and decomposed[0].isascii() and decomposed[0].isalpha():
        return decomposed[0]
    return char


def fix_lookalikes(text: str) -> str:
    """
    Repair mixed-script token corruption in a model answer.

    The local model occasionally emits a Cyrillic or accented Latin
    lookalike inside an otherwise Latin word (``tеhnologіje``,
    ``Računarské``).  Serbian transliteration leaves those characters in
    place, so they are mapped to their plain Latin equivalents here.
    """
    if not text:
        return text
    chars: list[str] = []
    for char in text:
        if char in _CYRILLIC_LOOKALIKES:
            chars.append(_CYRILLIC_LOOKALIKES[char])
        elif char.isalpha() and not char.isascii():
            chars.append(_strip_latin_accent(char))
        else:
            chars.append(char)
    return "".join(chars)


def has_diacritic(word: str) -> bool:
    return any(char in _DIACRITICS for char in word)


def _restore_case(source: str, target: str) -> str:
    """Apply the source word's capitalisation to a lowercase target."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _outside_urls(text: str, transform) -> str:
    """Apply ``transform`` only to the text outside URLs/e-mail addresses."""
    parts: list[str] = []
    last = 0
    for match in _URL_RE.finditer(text):
        parts.append(transform(text[last:match.start()]))
        parts.append(match.group(0))
        last = match.end()
    parts.append(transform(text[last:]))
    return "".join(parts)


def _could_be_context_intro(text: str) -> bool:
    """True while ``text`` can still grow into a context-reference intro."""
    value = fold_word(text.strip())
    if not value:
        return True
    for phrase in _OPENING_PHRASES:
        folded = fold_word(phrase)
        if folded.startswith(value) or value.startswith(folded):
            return True
    return False


def build_diacritic_map(texts) -> dict[str, str]:
    """
    Map folded words to the single accented form found in ``texts``.

    A folded key that occurs in more than one spelling (``sto`` and
    ``što``) is skipped: the correct one cannot be chosen without context.
    Words shorter than four letters are ignored as too ambiguous, and words
    that are already unaccented need no correction.
    """
    forms: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        if not text:
            continue
        # URLs are not Serbian words; strip them before extracting tokens.
        for segment in _URL_RE.sub(" ", text).splitlines():
            for match in _WORD_RE.finditer(segment):
                word = match.group(0)
                if len(word) < 4 or not word.isalpha():
                    continue
                forms[fold_word(word)].add(word.lower())

    mapping: dict[str, str] = {}
    for key, variants in forms.items():
        if len(variants) != 1:
            continue  # ambiguous: both spellings occur in the context
        variant = next(iter(variants))
        if variant != key and has_diacritic(variant):
            mapping[key] = variant
    return mapping


def restore_diacritics(text: str, mapping: dict[str, str]) -> str:
    """Restore missing/wrong diacritics using a context-derived map."""
    if not mapping or not text:
        return text

    def transform(segment: str) -> str:
        def repl(match: re.Match) -> str:
            word = match.group(0)
            if len(word) < 4 or not word.isalpha() or has_diacritic(word):
                return word
            target = mapping.get(fold_word(word))
            if not target:
                return word
            return _restore_case(word, target)

        return _WORD_RE.sub(repl, segment)

    return _outside_urls(text, transform)


# Mistakes that have a single, unambiguous correct form.  Keep this list
# small and specific: the general spelling work is done by the prompt and
# the context-derived diacritic map, not by a hand-written dictionary.
_CORRECTIONS = (
    # ESPB is "evropski sistem prenosa bodova"; Qwen sometimes invents
    # "sistem za prepoznavanje kvalifikacija" or "prenosivi bodovi".
    # Replace only the phrase after "sistem", so its case ending survives.
    (re.compile(r"\bza\s+prepoznavanje\s+kvalifikacija\b", re.IGNORECASE),
     "prenosa bodova"),
    (re.compile(r"\bprenosivih\s+bodova\b", re.IGNORECASE),
     "prenosa bodova"),
    # A number followed by the wrong expansion ("240 evropskih sistema
    # prenosa bodova") reads best as the unit itself.
    (re.compile(r"\b(\d+)\s+evropsk\w*\s+sistem\w*\s+prenosa\s+bodova\b",
                re.IGNORECASE), r"\1 ESPB bodova"),
    (re.compile(r"\binformativne\s+tehnologije\b", re.IGNORECASE),
     "informacione tehnologije"),
    (re.compile(r"\binformativnog\s+servisa\b", re.IGNORECASE),
     "informacionog servisa"),
    (re.compile(r"\binformativni\s+servis\b", re.IGNORECASE),
     "informacioni servis"),
    # Croatian/ijekavian forms the model sometimes picks up.
    (re.compile(r"\bsmjer(\w*)\b", re.IGNORECASE), r"smer\1"),
    (re.compile(r"\bčetverogodišnj(\w*)\b", re.IGNORECASE),
     r"četvorogodišnj\1"),
    (re.compile(r"\bstudijskoga\b", re.IGNORECASE), "studijskog"),
    (re.compile(r"\binformacijsk(\w*)\b", re.IGNORECASE),
     r"informacion\1"),
    # Common misspellings / token glitches observed from the local model.
    (re.compile(r"\bnajkasnje\b", re.IGNORECASE), "najkasnije"),
    (re.compile(r"\bstudiama\b", re.IGNORECASE), "studijama"),
    (re.compile(r"\bprijemi\s+ispit\b", re.IGNORECASE), "prijemni ispit"),
    (re.compile(r"\bprijmnih\b", re.IGNORECASE), "prijemnih"),
    (re.compile(r"\bosnivnih\b", re.IGNORECASE), "osnovnih"),
    (re.compile(r"\bisprituji\s+se\s+preko\b", re.IGNORECASE),
     "ispiti se prijavljuju preko"),
    (re.compile(r"\bisprituji\b", re.IGNORECASE), "ispiti"),
    (re.compile(r"\bmogu\s+navedi\b", re.IGNORECASE), "mogu navesti"),
    (re.compile(r"\bjes\s+to\b", re.IGNORECASE), "jeste"),
    (re.compile(r"\bkao\s+well\s+kao\b", re.IGNORECASE), "kao i"),
    # A token-level glitch we have seen in streamed answers.
    (re.compile(r"\bizoror\b", re.IGNORECASE), "izvor"),
)


def apply_terminology_corrections(text: str) -> str:
    def replace(match: re.Match, replacement: str) -> str:
        return _restore_case(match.group(0), match.expand(replacement))

    for pattern, replacement in _CORRECTIONS:
        text = pattern.sub(
            lambda match, repl=replacement: replace(match, repl),
            text,
        )
    return text


def normalize_citations(text: str) -> str:
    """Use the app's ``[Izvor N]`` citation form, not a bare ``[N]``."""
    text = _DOUBLE_CITATION_RE.sub(r"[\1]", text)
    text = _BARE_CITATION_RE.sub(r"[Izvor \1]", text)
    return _CITATION_CASE_RE.sub(r"[Izvor \1]", text)


def strip_context_references(text: str) -> str:
    """
    Remove boilerplate about *where* the answer came from.

    The model sometimes opens with "Na osnovu priloženog konteksta, …" or
    "Prema dokumentaciji, …".  The answer should start with the answer
    itself, so these intros are dropped (and the next word capitalised).
    A short no-information answer phrased around the context is replaced by
    the direct "Nemam informaciju o tome."; longer answers with extra
    content are left alone rather than losing it.
    """
    if not text:
        return text

    # Whole short answer is only a statement about the context.
    if (len(text) <= _MAX_CONTEXT_ONLY_ANSWER
            and not _CITATION_CASE_RE.search(text)
            and not _BARE_CITATION_RE.search(text)
            and _CONTEXT_NO_INFO_RE.match(text)
            and len(re.findall(r"[.!?](?:\s|$)", text)) <= 1
            and not re.search(r"(?i)\b(?:ali|već|osim)\b", text)):
        return "Nemam informaciju o tome."

    # "Nemam informaciju o X jer kontekst ne sadrži …" — keep the direct
    # no-information answer, drop the explanation about the context.
    if (len(text) <= _MAX_CONTEXT_ONLY_ANSWER
            and not _CITATION_CASE_RE.search(text)
            and not _BARE_CITATION_RE.search(text)
            and re.match(r"(?i)^\s*nemam\s+informacij", text)
            and re.search(
                r"(?i)\b(?:jer|zato\s+što|pošto)\s+(?:u\s+)?"
                r"(?:prilo[žz]en\w*\s+)?(?:kontekst\w*|dokumentacij\w*)",
                text,
            )):
        return "Nemam informaciju o tome."

    text = _CONTEXT_INTRO_RE.sub(
        lambda match: match.group(1) + match.group(2).upper(), text,
    )
    text = _CONTEXT_PARENTHETICAL_RE.sub("", text)
    text = _CONTEXT_CLAIM_RE.sub(
        lambda match: match.group(1) + match.group(2).upper(), text,
    )
    text = _MULTI_SPACE_RE.sub(" ", text)
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def clean_text(text: str) -> str:
    """Normalise Unicode and remove invisible/duplicated whitespace."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


class SerbianAnswerNormalizer:
    """
    Normalise answer text to correct Serbian (Latin by default).

    ``context_texts`` are the retrieved chunks; their vocabulary is used to
    restore dropped diacritics.  ``script`` may be ``"latinic"`` (default)
    or ``"cyrillic"``.
    """

    def __init__(self, translator, context_texts=(), script: str = "latinic"):
        self.translator = translator
        self.script = normalize_script(script)
        self._mapping = build_diacritic_map(
            self._to_latinic(text) for text in context_texts if text
        )
        self._pending = ""
        self._started = False
        self._opening_suspect = False

    # -- translator compatibility -------------------------------------- #

    def _to_latinic(self, text: str) -> str:
        for name in ("to_latinic", "cyrillic_to_latinic", "normalize_query"):
            converter = getattr(self.translator, name, None)
            if callable(converter):
                converted = converter(text)
                return converted if isinstance(converted, str) else text
        return text

    def _to_script(self, text: str) -> str:
        converter = getattr(self.translator, "to_script", None)
        if callable(converter):
            converted = converter(text, self.script)
            if isinstance(converted, str):
                return converted
        if self.script == "cyrillic":
            converter = getattr(self.translator, "latinic_to_cyrillic", None)
            if callable(converter):
                converted = converter(text)
                return converted if isinstance(converted, str) else text
        return text

    # -- normalisation ------------------------------------------------- #

    def normalize(self, text: str) -> str:
        """Normalise a complete piece of text (non-streaming path)."""
        if not text:
            return text
        # URLs/e-mails are opaque: transliteration or corrections inside
        # them would break links.
        text = _outside_urls(text, self._to_latinic)
        text = _outside_urls(text, fix_lookalikes)
        text = clean_text(text)
        text = strip_context_references(text)
        text = restore_diacritics(text, self._mapping)
        text = apply_terminology_corrections(text)
        text = normalize_citations(text)
        return _outside_urls(text, self._to_script)

    def feed(self, token: str) -> str:
        """
        Add one streaming token and return whatever is complete enough to
        emit.  The trailing word is held back until its boundary arrives,
        so a word is never split across output chunks.
        """
        if not token:
            return ""
        self._pending += token

        # Hold the opening until we know whether it is a context intro.
        if not self._started:
            if self._hold_opening():
                self._opening_suspect = True
                return ""
            self._started = True
            if self._opening_suspect:
                # The opening was suspicious and may now be complete.  Strip
                # it here, before the trailing word is emitted; the normal
                # per-chunk normalisation below cannot see the full phrase.
                self._pending = strip_context_references(self._pending)
                self._opening_suspect = False

        trailing_word = _TRAILING_WORD_RE.search(self._pending)
        if trailing_word:
            cut = trailing_word.start()
        else:
            # Keep trailing whitespace too, so "word " + "," becomes
            # "word," instead of "word ,".
            cut = len(self._pending.rstrip())

        # A citation can be split across tokens ("[", "1", "]"); hold it
        # until the closing bracket arrives so it is normalised as a whole.
        # The hold is bounded: an unclosed "[" must not stall the stream.
        bracket = self._pending.rfind("[")
        if (bracket != -1
                and len(self._pending) - bracket <= _MAX_CITATION_HOLD
                and "]" not in self._pending[bracket:]):
            cut = min(cut, bracket)

        if cut <= 0:
            return ""
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return self.normalize(ready)

    def _hold_opening(self) -> bool:
        """
        Should the first tokens be held while an intro is still ambiguous?

        A comma/colon followed by the next word is enough to tell whether
        the opening was "Na osnovu priloženog konteksta, …" (strip it) or a
        normal sentence (keep it).  "…da program…" is enough for
        "Kontekst navodi da…".  The hold is bounded so an unusual opening
        cannot stall the stream.
        """
        if re.search(r"(?i)[,:;]\s*[^\W\d_]", self._pending):
            return False
        if re.search(r"(?i)\bda\s+[^\W\d_]", self._pending):
            return False
        if "\n" in self._pending or len(self._pending) >= _OPENING_HOLD_MAX:
            return False
        return _could_be_context_intro(self._pending)

    def flush(self) -> str:
        """Emit the last held-back word (end of stream)."""
        self._started = True
        ready, self._pending = self._pending, ""
        return self.normalize(ready)
