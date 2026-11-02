"""Regressions for text preservation and CLI input handling; no model download."""
import unittest

from mark_stress import transfer_marks
from synthesize import split_text


class TextTests(unittest.TestCase):
    def test_punctuation_numbers_latin_and_case_survive(self):
        self.assertEqual(
            transfer_marks("Договор: API, 12% — обновлён!", "догов+ор обновлён"),
            "Догово́р: API, 12% — обновлён!",
        )

    def test_dropped_words_are_kept(self):
        self.assertEqual(transfer_marks("Потом закрыла дверь.", "закр+ыла дверь"),
                         "Потом закры́ла дверь.")

    def test_manual_stress_is_not_overridden(self):
        self.assertEqual(transfer_marks("За́мок и замок.", "зам+ок и з+амок"),
                         "За́мок и за́мок.")

    def test_yo_and_single_vowels_are_unmarked(self):
        self.assertEqual(transfer_marks("Елка и дом.", "ёлка +и д+ом"), "Ёлка и дом.")

    def test_splitting_keeps_every_word(self):
        source = "Первая фраза. " + "Очень длинный текст для проверки. " * 20
        chunks = split_text(source, limit=35)
        self.assertEqual(" ".join(chunks).split(), source.split())
        self.assertTrue(all(len(piece) <= 35 for piece in chunks))

    def test_acute_does_not_get_separated_from_vowel(self):
        self.assertEqual(split_text("За́мок стоит.\nЗамо́к закрыт."),
                         ["За́мок стоит.", "Замо́к закрыт."])


if __name__ == "__main__":
    unittest.main()
