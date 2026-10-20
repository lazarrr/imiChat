"""
keyword_search.py
~~~~~~~~~~~~~~~~~
Keyword side of retrieval, used by VectorStoreManager when switched on:

  HYBRID_SEARCH   BM25 keyword search fused with the vector search.  The
                  E5 embeddings capture meaning but are weak at exact
                  words: course names, professors, rooms, codes, years.
  QUERY_SYNONYMS  adds the official terms for student wording before
                  searching (smer → modul, OAS → osnovne akademske
                  studije), from config/synonyms.txt.

Serbian is highly inflected, so words are compared after folding (lower
case, Cyrillic → Latin, no diacritics) and light stemming that cuts common
case endings — "predmet", "predmeta" and "predmetima" all match.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

_CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "dj", "е": "e",
    "ж": "z", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "c", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "c",
    "џ": "dz", "ш": "s",
})

_STOPWORDS = frozenset("""
    a ako ali bi bih bio bila bile bili bilo biti ce cu da do gde i ih ili im
    ima imaju iz ja je jer jos joj ju kad kada kako kao ko koja koje koji kojih
    kojim kojima koju kolika koliki koliko li ma me mi mogu moze mu na nad nam
    nas ne nego nema neki nesto ni nije niti njih njima o od on ona one oni
    ono pa po pod pre preko pri sa sam se si smo sta ste su sve svi ta tako te
    ti to treba tu u uz vam vas za zasto
""".split())

# Common Serbian case / number endings, longest first ("predmetima" → "predmet")
_SUFFIXES = (
    "ovima", "evima", "ama", "ima", "ovi", "ova", "ove", "evi", "eva", "eve",
    "om", "em", "og", "eg", "ih", "im", "oj", "a", "e", "i", "o", "u",
)
_MIN_STEM = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower case, Cyrillic → Latin, no diacritics ("Šta je ESPB?" → "sta je espb?")."""
    text = text.casefold().translate(_CYRILLIC_TO_LATIN).replace("đ", "dj")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Folded, stemmed words without stopwords — the units BM25 counts."""
    return [_stem(w) for w in _WORD_RE.findall(fold(text)) if w not in _STOPWORDS]


class BM25Index:
    """
    Okapi BM25 over a fixed list of chunks (k1=1.5, b=0.75).  Only chunks
    sharing at least one word with the query get a score.
    """

    def __init__(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.ids, self.documents, self.metadatas = ids, documents, metadatas
        self.k1, self.b = k1, b
        self._postings: dict[str, list[tuple[int, int]]] = {}  # word → [(chunk, count)]
        self._lengths: list[int] = []
        for i, doc in enumerate(documents):
            words = tokenize(doc or "")
            self._lengths.append(len(words))
            for word, count in Counter(words).items():
                self._postings.setdefault(word, []).append((i, count))
        n = len(documents)
        self._avg_length = sum(self._lengths) / n if n else 0.0
        self._idf = {
            word: math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for word, postings in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self,
        query: str,
        n: int,
        allow: Callable[[dict], bool] | None = None,
    ) -> list[tuple[str, str, dict]]:
        """Best ``n`` chunks as (id, document, metadata); ``allow`` filters by metadata."""
        scores: dict[int, float] = {}
        for word in set(tokenize(query)):
            idf = self._idf.get(word)
            if idf is None:
                continue
            for i, count in self._postings[word]:
                norm = count + self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_length)
                scores[i] = scores.get(i, 0.0) + idf * count * (self.k1 + 1) / norm

        hits: list[tuple[str, str, dict]] = []
        for i in sorted(scores, key=scores.get, reverse=True):
            meta = self.metadatas[i] or {}
            if allow is None or allow(meta):
                hits.append((self.ids[i], self.documents[i], meta))
                if len(hits) >= n:
                    break
        return hits


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60) -> list[str]:
    """
    Merge ranked id lists: each id scores sum(1 / (k + rank)) over the lists
    it appears in, so ids ranked well by both searches come first.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class SynonymExpander:
    """
    Adds the other terms of a synonym group to a query that contains one of
    them, so student wording also finds documents written with the official
    term:  "Koji su smerovi na informatici?" → "... (modul)".
    """

    def __init__(self, groups: Iterable[Iterable[str]]):
        # each group: [(term as written, its folded + stemmed words), ...]
        self.groups: list[list[tuple[str, frozenset[str]]]] = []
        for group in groups:
            entries = [(term, frozenset(tokenize(term))) for term in group]
            self.groups.append([(term, words) for term, words in entries if words])

    @classmethod
    def from_file(cls, path: str | Path) -> SynonymExpander:
        """One comma-separated group per line; '#' starts a comment."""
        groups = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            terms = [t.strip() for t in line.split("#", 1)[0].split(",") if t.strip()]
            if len(terms) > 1:
                groups.append(terms)
        return cls(groups)

    def expand(self, query: str) -> str:
        words = set(tokenize(query))
        additions: list[str] = []
        for group in self.groups:
            if any(term_words <= words for _, term_words in group):
                additions += [
                    term for term, term_words in group
                    if not term_words <= words and term not in additions
                ]
        return f"{query} ({', '.join(additions)})" if additions else query
