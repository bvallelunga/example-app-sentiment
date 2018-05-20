import unittest

from app.main import MAX_CHAR_COUNT
from app.main import MAX_TEXT_COUNT
from app.main import ModelInterface
from app.main import SCORE_PRECISION


class TestModelInterface(unittest.TestCase):
    def setUp(self):
        self.interface = ModelInterface()

    def tearDown(self):
        self.interface = None

    def test_input_is_missing_texts_key(self):
        """'texts' is a required key; raise KeyError if 'texts' not present."""
        with self.assertRaises(KeyError):
            self.interface.prediction({})

    def test_empty_list_of_texts(self):
        """'texts' argument can not be empty; raise ValueError if 'texts' is an empty list."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'texts': []})

    def test_number_of_texts_above_max(self):
        """Text count can not exceed the max; raise ValueError if 'texts' exceeds max text count."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'texts': ['hello world'] * (MAX_TEXT_COUNT + 1)})

    def test_text_character_count_above_max(self):
        """Character count, of any text, can not exceed the max; raise ValueError if character count exceeds max
        character count."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'texts': ['hello world', 'a' * (MAX_CHAR_COUNT + 1)]})

    def test_texts_is_not_list(self):
        """'texts' must be a list; raise Value error if 'texts' is not a list."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'texts': set()})

    def test_texts_are_not_strings(self):
        """Texts, in 'texts', must be strings; raise ValueError if texts in 'texts' are not strings."""
        with self.assertRaises(ValueError):
            self.interface.prediction({'texts': ['hello world', 1, 2.2]})

    def test_texts_key_in_output(self):
        """The 'texts' key must be in the output."""
        self.assertTrue('texts' in self.interface.prediction({'texts': ['hello world']}))

    def test_input_texts_are_in_output(self):
        """All texts must be in the output."""
        input_texts = ['hello world', 'foobar for days']
        output_texts = self.interface.prediction({'texts': input_texts})['texts']
        self.assertTrue(all(text in output_texts for text in input_texts))

    def test_texts_in_output_are_strings(self):
        """All words in the output must be strings."""
        texts = self.interface.prediction({'texts': ['cat']})['texts']
        self.assertTrue(all(isinstance(text, str) for text in texts))

    def test_texts_have_score_key(self):
        """All texts must have a score key."""
        texts = self.interface.prediction({'texts': ['hello world', 'foobar for days']})['texts']
        self.assertTrue(all('score' in text for text in texts.values()))

    def test_scores_are_floats(self):
        """All scores must be floats."""
        texts = self.interface.prediction({'texts': ['hello world', 'foobar for days']})['texts']
        self.assertTrue(all(isinstance(text['score'], float) for text in texts.values()))

    def test_scores_have_correct_precision(self):
        """All scores must have the correct precision."""
        texts = self.interface.prediction({'texts': ['hello world', 'foobar for days']})['texts']
        self.assertTrue(all(text['score'] == round(text['score'], ndigits=SCORE_PRECISION) for text in texts.values()))


if __name__ == '__main__':
    unittest.main()
