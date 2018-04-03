from encoder import Model
from utils import load_model_params

MODEL_PARAMS_PATH = 'model'
SENTIMENT_NEURON_IDX = 2388
SCORE_PRECISION = 2
MAX_TEXT_COUNT = 10
MAX_CHAR_COUNT = 1500


class ModelInterface(object):
    def __init__(self, params=None):
        if params is None:
            params = load_model_params(MODEL_PARAMS_PATH)
        self.model = Model(params)  # very slow

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

        scores = self.model.transform(input['texts'])[:, SENTIMENT_NEURON_IDX]
        results = {input['texts'][i]: {'score': round(scores[i].item(), ndigits=SCORE_PRECISION)}
                   for i in range(len(scores))}
        return {'texts': results}


if __name__ == '__main__':
    interface = ModelInterface()
    print(interface.prediction({"texts": ["What a beautiful product.", "The color is very dull."]}))
