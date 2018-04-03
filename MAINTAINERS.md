# Sentiment
This app uses the "sentiment neuron" found in the OpenAI byte-level language model described in "
Learning to Generate Reviews and Discovering Sentiment". 


## Model Background
The model code and weights, from the paper, are open sourced. Specifically, 
we make use of the `encoder.py` and `utils.py` files. `encoder.py` is modified such that
the model instantiation and weight loading are decoupled allowing for faster unit testing. 
To achieve this, we also added a `load_model_params()` function to `utils.py`.


## Resources
  * [OpenAI model repo](https://github.com/openai/generating-reviews-discovering-sentiment)
  * [OpenAI model paper](https://arxiv.org/abs/1704.01444)
  * [Model Depot example](https://modeldepot.io/afowler/sentiment-neuron/overview)
  * [Sentiment Neuron blog post](https://rakeshchada.github.io/Sentiment-Neuron.html)

