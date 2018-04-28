# Sentiment

## App Design
This app uses the "sentiment neuron" found in the byte-level language model described in OpenAI's paper, ["Learning to Generate 
Reviews and Discovering Sentiment"][1]. The model weights and TensorFlow implementation can be found at the paper's 
accompanying github repo, [openai/generating-reviews-discovering][2].

The model is trained on the Amazon product review dataset, a dataset of over 82 million Amazon product reviews.

## Contributing
Code should be written for Python 3, include documentation (docstrings & comments), follow PEP 8 and pass all unittests.
To run the unittests, simply run `python -m unittest` from the repo directory.


[1]: https://arxiv.org/abs/1704.01444
[2]: https://github.com/openai/generating-reviews-discovering-sentiment