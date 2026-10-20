"""
version_filter.py
~~~~~~~~~~~~~~~~~
Which study-program version does a question ask about?

The scraper (imi-scraper) labels every chunk from a program page with

    version_label        "OAS Informatika — akreditacija 2019"
    accreditation_year   2019
    status               enrolling | phase_out | upcoming | legacy

and leaves both fields off version-agnostic content (FAQ, general info).
The statuses mean:

    enrolling  current program, enrollment still open        → current
    phase_out  old program, earlier students still study it  → historical
    upcoming   new program, enrollment has not started yet   → future
    legacy     program that is no longer studied             → historical

This module turns the question into a ``VersionIntent``:

    no year / "aktuelni program"       → current (enrolling + neutral chunks)
    "2019", "akreditacija 2011"        → historical versions of that year
    "stari program", "prethodni upis"  → every legacy / phase_out version
    "2027", "upis 2027/28"             → upcoming versions of that year
    "budući program", "sledeće godine" → every upcoming version

The intent provides both the Chroma metadata filter (``where()``) and the
Python-side predicate used after retrieval (``matches()``), so the keyword
search and the reranked candidates are filtered by exactly the same rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

# Status values written by the scraper's programs.yaml registry.
STATUS_ENROLLING = "enrolling"
STATUS_PHASE_OUT = "phase_out"
STATUS_UPCOMING = "upcoming"
STATUS_LEGACY = "legacy"

OLD_STATUSES = frozenset({STATUS_LEGACY, STATUS_PHASE_OUT})
FUTURE_STATUSES = frozenset({STATUS_UPCOMING})
CURRENT_STATUSES = frozenset({STATUS_ENROLLING})

STATUS_LABEL = {
    STATUS_ENROLLING: "aktuelni program (upis u toku)",
    STATUS_PHASE_OUT: "prethodni program (važi za ranije upisane studente)",
    STATUS_UPCOMING: "novi program (upis još nije počeo)",
    STATUS_LEGACY: "stari program",
}

# Intent modes.
MODE_CURRENT = "current"
MODE_HISTORICAL = "historical"
MODE_UPCOMING = "upcoming"
MODE_YEARS = "years"  # several explicit years, e.g. "2019 i 2027"

# 2000–2099, not part of a longer number ("2019/20" still yields 2019).
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

# "stari program", "po starom planu", "prethodna akreditacija",
# "ranije upisani studenti", "bivši program".
_OLD_QUERY_RE = re.compile(
    r"\bstar(?:i|og|om|im|ih|e|a|u)\b[^.!?\n]{0,30}\b(?:program\w*|akreditacij\w*|plan\w*)"
    r"|\bprethodn(?:i|og|om|im|e|a|u)\b[^.!?\n]{0,30}\b(?:program\w*|akreditacij\w*|upis\w*)"
    r"|\bpo\s+starom\b"
    r"|\branije\s+upisan\w*"
    r"|\bbiv(?:ši|šeg|šem|še|ša)\b[^.!?\n]{0,30}\bprogram\w*",
    re.IGNORECASE,
)

# Only unambiguous future wording.  "novi program" is deliberately NOT here:
# in the current accreditation cycle it usually means the program that is
# enrolling now, not the one after it.  An explicit future year (2027) is
# what selects the upcoming status.
_FUTURE_QUERY_RE = re.compile(
    r"\bbuduć(?:i|eg|em|e|a|u)\b[^.!?\n]{0,30}\b(?:program\w*|akreditacij\w*)"
    r"|\b(?:naredn|sledeć)(?:e|oj|u|im|ih)\s+(?:godin\w*|upis\w*)",
    re.IGNORECASE,
)

# Applied inside the Chroma query.  $ne also keeps documents that do not
# have the key at all, which is what admits version-agnostic chunks.
CURRENT_ONLY_WHERE = {
    "$and": [
        {"status": {"$ne": STATUS_LEGACY}},
        {"status": {"$ne": STATUS_PHASE_OUT}},
        {"status": {"$ne": STATUS_UPCOMING}},
        {"outdated": {"$ne": True}},
    ]
}
OLD_ONLY_WHERE = {
    "$and": [
        {"$or": [
            {"status": {"$eq": STATUS_LEGACY}},
            {"status": {"$eq": STATUS_PHASE_OUT}},
        ]},
        {"outdated": {"$ne": True}},
    ]
}
UPCOMING_ONLY_WHERE = {
    "$and": [
        {"status": {"$eq": STATUS_UPCOMING}},
        {"outdated": {"$ne": True}},
    ]
}


def _years_where(years: tuple[int, ...]) -> dict:
    return {
        "$and": [
            {"accreditation_year": {"$in": list(years)}},
            {"outdated": {"$ne": True}},
        ]
    }


def enrollment_year(today: date | None = None) -> int:
    """Current enrollment year — matches the scraper's registry.

    Enrollment for academic year Y/Y+1 starts in June of year Y, so in
    October 2026 the current enrollment year is 2026.
    """
    today = today or date.today()
    return today.year if today.month >= 6 else today.year - 1


def _as_year(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _meta_year(meta: dict) -> int | None:
    """The chunk's accreditation year, falling back to the version label.

    ``accreditation_year`` is the authoritative field; older or partially
    migrated indexes only have ``version_label`` ("… — akreditacija 2019").
    """
    year = _as_year(meta.get("accreditation_year"))
    if year is not None:
        return year
    label_years = [int(y) for y in _YEAR_RE.findall(str(meta.get("version_label") or ""))]
    return max(label_years) if label_years else None


def _valid_for_contains(meta: dict, year: int) -> bool:
    """True if ``year`` falls inside the chunk's ``valid_for`` range.

    ``valid_for`` is text such as "studenti upisani 2019/20–2025/26" or
    "upis od 2027/28".  It lets a question about the 2025 enrollment match
    the version accredited in 2019 even though 2025 is not its
    accreditation year.
    """
    years = [int(y) for y in _YEAR_RE.findall(str(meta.get("valid_for") or ""))]
    if not years:
        return False
    start, end = min(years), max(years)
    # "2025/26" ends in June 2026, so the last year is end + 1.
    return start <= year <= end + 1


def _family(status, year: int | None, current_year: int) -> str | None:
    """Coarse version family: current / old / future / None (agnostic)."""
    status = str(status or "").strip().lower()
    if status in OLD_STATUSES:
        return "old"
    if status in FUTURE_STATUSES:
        return "future"
    if status in CURRENT_STATUSES:
        return "current"
    if year is None:
        return None
    if year < current_year:
        return "old"
    if year > current_year:
        return "future"
    return "current"


@dataclass(frozen=True)
class VersionIntent:
    """Which program version(s) a question asks for."""

    mode: str = MODE_CURRENT
    years: tuple[int, ...] = ()
    current_year: int = field(default_factory=enrollment_year)

    # ------------------------------------------------------------------ #
    #  Question → intent                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_query(cls, query: str, today: date | None = None) -> "VersionIntent":
        current = enrollment_year(today)
        text = query or ""
        years = tuple(dict.fromkeys(int(y) for y in _YEAR_RE.findall(text)))

        if years:
            older = tuple(y for y in years if y < current)
            newer = tuple(y for y in years if y > current)
            if not older and not newer:
                mode = MODE_CURRENT
            elif older and not newer:
                mode = MODE_HISTORICAL
            elif newer and not older:
                mode = MODE_UPCOMING
            else:
                mode = MODE_YEARS
            return cls(mode=mode, years=years, current_year=current)

        if _OLD_QUERY_RE.search(text):
            return cls(mode=MODE_HISTORICAL, current_year=current)
        if _FUTURE_QUERY_RE.search(text):
            return cls(mode=MODE_UPCOMING, current_year=current)
        return cls(mode=MODE_CURRENT, current_year=current)

    @classmethod
    def current(cls, today: date | None = None) -> "VersionIntent":
        return cls(mode=MODE_CURRENT, current_year=enrollment_year(today))

    # ------------------------------------------------------------------ #
    #  Chroma filter                                                       #
    # ------------------------------------------------------------------ #

    def where(self) -> dict | None:
        """Primary metadata filter for the Chroma query."""
        if self.mode == MODE_CURRENT:
            return CURRENT_ONLY_WHERE
        if self.mode == MODE_HISTORICAL:
            return _years_where(self.years) if self.years else OLD_ONLY_WHERE
        if self.mode == MODE_UPCOMING:
            return _years_where(self.years) if self.years else UPCOMING_ONLY_WHERE
        return _years_where(self.years)  # MODE_YEARS

    def fallback_where(self) -> dict | None:
        """Broader filter tried only when the primary one finds nothing.

        An older index has no version metadata at all, and an explicit year
        in the question may be an enrollment year rather than an
        accreditation year.  Falling back to the version family (and then to
        no filter) keeps such questions answerable; ``matches(relaxed=True)``
        is what decides which chunks from that wider set may pass.
        """
        if self.mode == MODE_HISTORICAL:
            return OLD_ONLY_WHERE
        if self.mode == MODE_UPCOMING:
            return UPCOMING_ONLY_WHERE
        return None

    # ------------------------------------------------------------------ #
    #  Python-side predicate (keyword search + reranked candidates)        #
    # ------------------------------------------------------------------ #

    def matches(self, meta: dict, relaxed: bool = False) -> bool:
        """True if this chunk belongs to the version(s) the question asks for.

        ``relaxed`` is used only for chunks fetched by ``fallback_where()``:
        it additionally accepts chunks without a year and chunks whose
        ``valid_for`` range covers a mentioned enrollment year.
        """
        meta = meta or {}
        if meta.get("outdated") is True:
            return False

        year = _meta_year(meta)
        family = _family(meta.get("status"), year, self.current_year)

        # Version-agnostic content (general info, FAQ) has no status/year.
        # It is always useful for a current question; for a year-specific
        # question it is admitted only by the relaxed fallback.
        if family is None:
            return relaxed or self.mode == MODE_CURRENT

        if self.mode == MODE_CURRENT:
            return family == "current"
        if self.mode == MODE_HISTORICAL:
            return family == "old" and self._matches_old(meta, year, relaxed)
        if self.mode == MODE_UPCOMING:
            return family == "future" and self._matches_future(meta, year, relaxed)
        return self._matches_years(meta, year, relaxed)

    def _matches_old(self, meta: dict, year: int | None, relaxed: bool) -> bool:
        if not self.years:
            return True
        if year in self.years:
            return True
        if not relaxed:
            return False
        if year is None:
            return True
        return any(_valid_for_contains(meta, wanted) for wanted in self.years)

    def _matches_future(self, meta: dict, year: int | None, relaxed: bool) -> bool:
        if not self.years:
            return True
        if year in self.years:
            return True
        if not relaxed:
            return False
        if year is None:
            return True
        return any(_valid_for_contains(meta, wanted) for wanted in self.years)

    def _matches_years(self, meta: dict, year: int | None, relaxed: bool) -> bool:
        if year in self.years:
            return True
        if not relaxed:
            return False
        if year is None:
            return True
        return any(_valid_for_contains(meta, wanted) for wanted in self.years)

    # ------------------------------------------------------------------ #
    #  Explanations                                                        #
    # ------------------------------------------------------------------ #

    def reason(self, meta: dict) -> str:
        """Diagnostic reason for a chunk this intent rejected."""
        meta = meta or {}
        if meta.get("outdated") is True:
            return "excluded_outdated"

        year = _meta_year(meta)
        family = _family(meta.get("status"), year, self.current_year)

        if self.years and year is not None and year not in self.years:
            return "excluded_version_year"
        if family == "future":
            return "excluded_upcoming_version"
        if family == "old":
            return "excluded_legacy_version"
        if family == "current":
            return "excluded_current_version"
        return "excluded_version_filter"

    def describe(self) -> str:
        """Short human-readable summary for logs."""
        if self.mode == MODE_CURRENT:
            return "current (enrolling + version-agnostic chunks)"
        if self.mode == MODE_HISTORICAL:
            wanted = ", ".join(str(y) for y in self.years)
            return f"historical ({wanted})" if wanted else "historical (legacy/phase_out)"
        if self.mode == MODE_UPCOMING:
            wanted = ", ".join(str(y) for y in self.years)
            return f"upcoming ({wanted})" if wanted else "upcoming (status=upcoming)"
        return "years " + ", ".join(str(y) for y in self.years)
