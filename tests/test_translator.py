"""
Tests for Serbian script conversion (src/processing/translator.py).

    python -m unittest tests.test_translator -v
"""

import unittest

from src.processing.translator import ScriptTranslator, normalize_script


class ScriptDetectionTests(unittest.TestCase):
    def setUp(self):
        self.translator = ScriptTranslator()

    def test_detects_serbian_cyrillic_letters_outside_the_russian_block(self):
        # ђ, ћ, џ, љ and њ are outside the Russian а-я range; a narrow
        # regex would call these strings Latin.
        for text in ("ђак", "ћутање", "џеп", "љубав", "његов"):
            with self.subTest(text=text):
                self.assertEqual(self.translator.detect_script(text), "cyrillic")

    def test_latin_and_empty_text_are_latinic(self):
        for text in ("Đak", "čćšžđ", "", "Uvod u programiranje"):
            with self.subTest(text=text):
                self.assertEqual(self.translator.detect_script(text), "latinic")


class TransliterationTests(unittest.TestCase):
    def setUp(self):
        self.translator = ScriptTranslator()

    def test_cyrillic_is_transliterated_with_diacritics(self):
        self.assertEqual(
            self.translator.to_latinic("Студије информатике трају четири године."),
            "Studije informatike traju četiri godine.",
        )
        self.assertEqual(
            self.translator.to_latinic("Љубав, Његош и Џорџ"),
            "Ljubav, Njegoš i Džordž",
        )

    def test_mixed_script_word_is_repaired(self):
        # This is the form Qwen produced when it drifted: a Cyrillic
        # prefix glued to a Latin word.
        self.assertEqual(
            self.translator.to_latinic("прiloženog konteksta"),
            "priloženog konteksta",
        )

    def test_latin_text_is_returned_unchanged(self):
        text = "Uvod u programiranje nosi 6 ESPB (č, ć, š, ž, đ)."
        self.assertEqual(self.translator.to_latinic(text), text)

    def test_to_script_accepts_the_documented_aliases(self):
        self.assertEqual(self.translator.to_script("čćšžđ", "latinic"), "čćšžđ")
        self.assertEqual(self.translator.to_script("čćšžđ", "latinc"), "čćšžđ")
        self.assertEqual(self.translator.to_script("čćšžđ", "cyrillic"), "чћшжђ")
        self.assertEqual(self.translator.to_script("чћшжђ", "latinic"), "čćšžđ")

    def test_normalize_script_defaults_to_latin(self):
        self.assertEqual(normalize_script(None), "latinic")
        self.assertEqual(normalize_script("unknown"), "latinic")
        self.assertEqual(normalize_script("ćirilica"), "cyrillic")


if __name__ == "__main__":
    unittest.main()
