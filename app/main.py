# Add Current Directory to Path
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SCORE_PRECISION = 2
MAX_TEXT_COUNT = 10
MAX_CHAR_COUNT = 1500


class ModelInterface(object):
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()

    @staticmethod
    def is_list_of_strs(input):
        return isinstance(input, list) and len(input) > 0 and all(isinstance(el, str) for el in input)

    def prediction(self, input):
        if 'texts' not in input:
            raise KeyError("Expected key named 'texts' in input.")
        if not self.is_list_of_strs(input['texts']):
            raise ValueError("'texts' should be a list of strings.")
        if len(input['texts']) > MAX_TEXT_COUNT:
            raise ValueError("Number of texts can not exceed {}.".format(MAX_TEXT_COUNT))
        if any(len(string) > MAX_CHAR_COUNT for string in input['texts']):
            raise ValueError("Number of characters per text can not exceed {}.".format(MAX_CHAR_COUNT))

        scores = [self.model.polarity_scores(text)['compound'] for text in input['texts']]
        results = {input['texts'][i]: {'score': round(scores[i], ndigits=SCORE_PRECISION)}
                   for i in range(len(scores))}
        return {'texts': results}
