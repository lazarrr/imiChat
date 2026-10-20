"""
Tests for the question → study-program-version policy
(src/core/version_filter.py).  Pure Python — no Chroma or model needed.

    python -m unittest tests.test_version_filter -v
"""

import unittest
from datetime import date

from src.core.version_filter import (
    CURRENT_ONLY_WHERE,
    MODE_CURRENT,
    MODE_HISTORICAL,
    MODE_UPCOMING,
    MODE_YEARS,
    OLD_ONLY_WHERE,
    UPCOMING_ONLY_WHERE,
    VersionIntent,
)

TODAY = date(2026, 10, 19)  # current enrollment year: 2026


def intent(question, today=TODAY):
    return VersionIntent.from_query(question, today=today)


class QuestionParsingTests(unittest.TestCase):
    def test_no_year_means_the_current_program(self):
        parsed = intent("Koliko modula ima informatika?")

        self.assertEqual(parsed.mode, MODE_CURRENT)
        self.assertEqual(parsed.current_year, 2026)
        self.assertEqual(parsed.where(), CURRENT_ONLY_WHERE)

    def test_the_current_year_is_still_the_current_program(self):
        self.assertEqual(intent("Koliki je upis 2026/27?").mode, MODE_CURRENT)

    def test_an_older_year_selects_the_historical_version(self):
        parsed = intent("Koliko modula je imala informatika 2019. godine?")

        self.assertEqual(parsed.mode, MODE_HISTORICAL)
        self.assertEqual(parsed.years, (2019,))
        self.assertEqual(parsed.where(), {
            "$and": [
                {"accreditation_year": {"$in": [2019]}},
                {"outdated": {"$ne": True}},
            ]
        })

    def test_a_future_year_selects_the_upcoming_version(self):
        parsed = intent("Koji moduli postoje za upis 2027/28?")

        self.assertEqual(parsed.mode, MODE_UPCOMING)
        self.assertEqual(parsed.years, (2027,))

    def test_a_generic_old_phrase_means_all_historical_versions(self):
        parsed = intent("Šta piše u starom programu?")

        self.assertEqual(parsed.mode, MODE_HISTORICAL)
        self.assertEqual(parsed.years, ())
        self.assertEqual(parsed.where(), OLD_ONLY_WHERE)

    def test_an_explicit_future_phrase_means_upcoming_versions(self):
        parsed = intent("Kakav je budući program informatike?")

        self.assertEqual(parsed.mode, MODE_UPCOMING)
        self.assertEqual(parsed.where(), UPCOMING_ONLY_WHERE)

    def test_a_plain_new_program_question_stays_on_the_current_program(self):
        # "novi program" in the current cycle usually means the program that
        # is enrolling now (2026), not a not-yet-started one.
        self.assertEqual(
            intent("Kada počinje novi master program informatike?").mode,
            MODE_CURRENT,
        )

    def test_years_from_both_sides_of_current_are_kept_together(self):
        parsed = intent("Koliko modula ima program iz 2019 i 2027?")

        self.assertEqual(parsed.mode, MODE_YEARS)
        self.assertEqual(parsed.years, (2019, 2027))


class MatchingTests(unittest.TestCase):
    def test_current_intent_keeps_enrolling_and_version_agnostic_chunks(self):
        parsed = intent("Koliko modula ima informatika?")

        self.assertTrue(parsed.matches({"status": "enrolling", "accreditation_year": 2026}))
        self.assertTrue(parsed.matches({}))
        self.assertFalse(parsed.matches({"status": "phase_out", "accreditation_year": 2019}))
        self.assertFalse(parsed.matches({"status": "legacy", "accreditation_year": 2011}))
        self.assertFalse(parsed.matches({"status": "upcoming", "accreditation_year": 2027}))
        self.assertFalse(parsed.matches({"outdated": True}))

    def test_an_explicit_old_year_matches_only_that_year(self):
        parsed = intent("Program iz 2019?")

        self.assertTrue(parsed.matches({"status": "phase_out", "accreditation_year": 2019}))
        self.assertFalse(parsed.matches({"status": "legacy", "accreditation_year": 2011}))
        self.assertFalse(parsed.matches({"status": "enrolling", "accreditation_year": 2026}))

    def test_the_version_label_supplies_the_year_when_the_field_is_missing(self):
        parsed = intent("Program iz 2019?")

        self.assertTrue(parsed.matches({
            "status": "legacy",
            "version_label": "OAS Informatika — akreditacija 2019",
        }))
        self.assertFalse(parsed.matches({
            "status": "legacy",
            "version_label": "OAS Informatika — akreditacija 2011",
        }))

    def test_a_generic_old_question_matches_legacy_and_phase_out(self):
        parsed = intent("Stari program?")

        self.assertTrue(parsed.matches({"status": "phase_out"}))
        self.assertTrue(parsed.matches({"status": "legacy"}))
        self.assertFalse(parsed.matches({"status": "enrolling"}))
        self.assertFalse(parsed.matches({"status": "upcoming"}))

    def test_a_future_year_matches_the_upcoming_status(self):
        parsed = intent("Upis 2027/28?")

        self.assertTrue(parsed.matches({"status": "upcoming", "accreditation_year": 2027}))
        self.assertFalse(parsed.matches({"status": "enrolling", "accreditation_year": 2026}))
        self.assertFalse(parsed.matches({"status": "upcoming", "accreditation_year": 2028}))

    def test_relaxed_matching_admits_a_chunk_without_version_fields(self):
        parsed = intent("Program iz 2019?")

        self.assertFalse(parsed.matches({}))
        self.assertTrue(parsed.matches({}, relaxed=True))

    def test_relaxed_matching_uses_the_valid_for_enrollment_range(self):
        parsed = intent("Koji program je važio za upis 2025?")
        chunk = {
            "status": "phase_out",
            "accreditation_year": 2019,
            "valid_for": "studenti upisani 2019/20–2025/26",
        }

        self.assertFalse(parsed.matches(chunk))  # 2025 is not the accreditation year
        self.assertTrue(parsed.matches(chunk, relaxed=True))

    def test_outdated_chunks_are_dropped_even_for_an_old_program(self):
        parsed = intent("Stari program informatike?")

        self.assertFalse(parsed.matches({"status": "legacy", "outdated": True}))
        self.assertFalse(parsed.matches({"outdated": True}, relaxed=True))

    def test_reasons_name_the_version_that_was_filtered(self):
        current = intent("Koliko modula ima informatika?")

        self.assertEqual(
            current.reason({"status": "legacy", "accreditation_year": 2011}),
            "excluded_legacy_version",
        )
        self.assertEqual(
            current.reason({"status": "phase_out", "accreditation_year": 2019}),
            "excluded_legacy_version",
        )
        self.assertEqual(
            current.reason({"status": "upcoming", "accreditation_year": 2027}),
            "excluded_upcoming_version",
        )
        self.assertEqual(current.reason({"outdated": True}), "excluded_outdated")

    def test_describe_names_the_status_for_logs(self):
        self.assertIn("enrolling", intent("Koliko modula?").describe())
        self.assertIn("legacy/phase_out", intent("Stari program?").describe())
        self.assertIn("upcoming", intent("Upis 2027?").describe())


if __name__ == "__main__":
    unittest.main()
