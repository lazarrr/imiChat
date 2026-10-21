"""
Tests for the deterministic Serbian answer normaliser
(src/processing/serbian_text.py).

    python -m unittest tests.test_serbian_text -v
"""

import unittest

from src.processing.serbian_text import SerbianAnswerNormalizer
from src.processing.translator import ScriptTranslator


def make_normalizer(context=(), script="latinic"):
    return SerbianAnswerNormalizer(ScriptTranslator(), context, script)


class ScriptTests(unittest.TestCase):
    def test_cyrillic_answer_is_transliterated(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Студије информатике трају четири године."),
            "Studije informatike traju četiri godine.",
        )

    def test_mixed_script_answer_is_repaired(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("На основу прiloženog konteksta."),
            "Na osnovu priloženog konteksta.",
        )

    def test_corrupted_token_lookalikes_are_repaired(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize(
                "Softversko inženjerstvo i informacionе tехnologіје, "
                "Rаčunarскé naukэ і Іnformacionе sístémе [Izvоr 1]."
            ),
            "Softversko inženjerstvo i informacione tehnologije, "
            "Računarske nauke i Informacione sisteme [Izvor 1].",
        )

    def test_requested_cyrillic_output(self):
        normalizer = make_normalizer(script="cyrillic")
        self.assertEqual(
            normalizer.normalize("Studije informatike traju četiri godine."),
            "Студије информатике трају четири године.",
        )
        self.assertEqual(
            normalizer.normalize("Vidi https://example.com/upis."),
            "Види https://example.com/upis.",
        )


class DiacriticRestorationTests(unittest.TestCase):
    def test_missing_diacritics_are_restored_from_context(self):
        normalizer = make_normalizer([
            "Modul Računarske nauke i Softversko inženjerstvo.",
        ])
        self.assertEqual(
            normalizer.normalize("Racunarske nauke su dostupne."),
            "Računarske nauke su dostupne.",
        )

    def test_cyrillic_context_supplies_latin_word_forms(self):
        normalizer = make_normalizer(["Рачунарске науке су модул."])
        self.assertEqual(
            normalizer.normalize("Racunarske nauke."),
            "Računarske nauke.",
        )

    def test_capitalisation_is_preserved(self):
        normalizer = make_normalizer(["Čačak"])
        self.assertEqual(normalizer.normalize("Cacak"), "Čačak")
        self.assertEqual(normalizer.normalize("CACAK"), "ČAČAK")

    def test_ambiguous_context_forms_are_not_guessed(self):
        # "sesti" (to sit) and "šesti" (sixth) fold to the same key; the
        # correct one cannot be chosen, so the text is left alone.
        normalizer = make_normalizer([
            "Sesti ispit je u ponedeljak.",
            "Šesti ispit je u utorak.",
        ])
        self.assertEqual(normalizer.normalize("Sesti ispit"), "Sesti ispit")

    def test_urls_and_emails_are_never_rewritten(self):
        normalizer = make_normalizer(["Čačak"])
        self.assertEqual(
            normalizer.normalize("Vidi https://example.com/cacak i Cacak."),
            "Vidi https://example.com/cacak i Čačak.",
        )
        self.assertEqual(
            normalizer.normalize("Pošalji na ime@example.com"),
            "Pošalji na ime@example.com",
        )


class TerminologyTests(unittest.TestCase):
    def test_espb_expansion_is_corrected(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Evropski sistem za prepoznavanje kvalifikacija"),
            "Evropski sistem prenosa bodova",
        )
        self.assertEqual(
            normalizer.normalize("240 evropskih sistema prenosivih bodova"),
            "240 ESPB bodova",
        )

    def test_official_module_terms_are_corrected(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Informativne tehnologije i informativni servis"),
            "Informacione tehnologije i informacioni servis",
        )

    def test_citations_use_the_izvor_marker(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Vidi [1] i [izvor 2]."),
            "Vidi [Izvor 1] i [Izvor 2].",
        )
        # A markdown link is not a citation.
        self.assertEqual(
            normalizer.normalize("[1](https://example.com)"),
            "[1](https://example.com)",
        )

    def test_punctuation_spacing_is_cleaned(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Zdravo ,  svete !"),
            "Zdravo, svete!",
        )

    def test_observed_model_misspellings_are_corrected(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Najkasnje pet dana."),
            "Najkasnije pet dana.",
        )
        self.assertEqual(
            normalizer.normalize("smjer Informatika, četverogodišnji program"),
            "smer Informatika, četvorogodišnji program",
        )
        self.assertEqual(
            normalizer.normalize("informacijskog servisa i mogu navedi"),
            "informacionog servisa i mogu navesti",
        )
        self.assertEqual(
            normalizer.normalize("Isprituji se preko servisa."),
            "Ispiti se prijavljuju preko servisa.",
        )

    def test_double_citations_are_collapsed(self):
        normalizer = make_normalizer()
        self.assertEqual(normalizer.normalize("[[Izvor 1]]"), "[Izvor 1]")
        self.assertEqual(normalizer.normalize("[[2]]"), "[Izvor 2]")


class DirectAnswerTests(unittest.TestCase):
    """Answers must start with the answer, not with where it came from."""

    def test_context_intro_is_removed(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize(
                "Na osnovu priloženog konteksta, studije traju četiri "
                "godine [Izvor 1]."
            ),
            "Studije traju četiri godine [Izvor 1].",
        )
        self.assertEqual(
            normalizer.normalize(
                "Prema priloženoj dokumentaciji, ispiti se prijavljuju "
                "preko servisa."
            ),
            "Ispiti se prijavljuju preko servisa.",
        )

    def test_context_parenthetical_is_removed(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize(
                "Program se, na osnovu priloženog konteksta, sastoji od tri "
                "modula."
            ),
            "Program se sastoji od tri modula.",
        )

    def test_context_claim_is_removed(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize(
                "Kontekst navodi da program traje četiri godine."
            ),
            "Program traje četiri godine.",
        )

    def test_context_only_no_info_answer_is_direct(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize("Kontekst ne sadrži podatke o ceni školarine."),
            "Nemam informaciju o tome.",
        )
        self.assertEqual(
            normalizer.normalize("U priloženom kontekstu nema informacija o tome."),
            "Nemam informaciju o tome.",
        )

    def test_no_info_explanation_about_context_is_removed(self):
        normalizer = make_normalizer()
        self.assertEqual(
            normalizer.normalize(
                "Nemam informaciju o tome koliko košta školarina jer kontekst "
                "ne sadrži podatke o ceni."
            ),
            "Nemam informaciju o tome.",
        )

    def test_longer_no_info_answer_is_not_truncated(self):
        normalizer = make_normalizer()
        text = "Kontekst ne sadrži podatke o ceni, ali upis počinje 1. jula."
        self.assertEqual(normalizer.normalize(text), text)

    def test_direct_answer_is_unchanged(self):
        normalizer = make_normalizer()
        text = "Studije traju četiri godine [Izvor 1]."
        self.assertEqual(normalizer.normalize(text), text)


class StreamingTests(unittest.TestCase):
    TEXT = "Računarske nauke su jedan modul. Studije traju četiri godine."
    CONTEXT = ["Računarske nauke"]

    def test_streamed_output_matches_whole_text(self):
        expected = make_normalizer(self.CONTEXT).normalize(self.TEXT)
        normalizer = make_normalizer(self.CONTEXT)

        tokens = ["Ra", "čunarske nauke su jedan modul. Stu",
                  "dije traju četiri godine."]
        streamed = "".join(normalizer.feed(token) for token in tokens)
        streamed += normalizer.flush()

        self.assertEqual(streamed, expected)

    def test_streamed_word_is_not_split_mid_word(self):
        normalizer = make_normalizer(self.CONTEXT)
        self.assertEqual(normalizer.feed("Racunar"), "")
        self.assertEqual(normalizer.feed("ske"), "")
        self.assertEqual(normalizer.feed(" nauke"), "Računarske ")
        self.assertEqual(normalizer.flush(), "nauke")

    def test_streamed_citation_is_not_split(self):
        normalizer = make_normalizer()
        streamed = normalizer.feed("Vidi [") + normalizer.feed("1") + normalizer.feed("].")
        streamed += normalizer.flush()
        self.assertEqual(streamed, "Vidi [Izvor 1].")

    def test_streamed_context_intro_is_removed(self):
        normalizer = make_normalizer()
        tokens = ["Na", " osnovu", " priloženog", " konteksta,", " studije",
                  " traju", " četiri", " godine."]
        streamed = "".join(normalizer.feed(token) for token in tokens)
        streamed += normalizer.flush()
        self.assertEqual(streamed, "Studije traju četiri godine.")

    def test_streamed_context_only_no_info_is_direct(self):
        normalizer = make_normalizer()
        tokens = ["Kontekst", " ne", " sadrži", " podatke", " o", " ceni."]
        streamed = "".join(normalizer.feed(token) for token in tokens)
        streamed += normalizer.flush()
        self.assertEqual(streamed, "Nemam informaciju o tome.")

    def test_streamed_lookalikes_are_repaired(self):
        normalizer = make_normalizer()
        tokens = ["tехnolog", "іје, Rаčunarскé", " naukэ"]
        streamed = "".join(normalizer.feed(token) for token in tokens)
        streamed += normalizer.flush()
        self.assertEqual(streamed, "tehnologije, Računarske nauke")

    def test_streamed_cyrillic_output_keeps_digraphs_whole(self):
        normalizer = make_normalizer(script="cyrillic")
        tokens = ["Љу", "бав и Ње", "гош"]
        self.assertEqual(
            "".join(normalizer.feed(token) for token in tokens)
            + normalizer.flush(),
            "Љубав и Његош",
        )


if __name__ == "__main__":
    unittest.main()
