# Sentiment

## App Design
This app uses VADER. VADER is lexicon and rule-based i.e. it is not
trained on any specific dataset. However, the model is "specifically
attuned to sentiments expressed in social media".

More information on VADER can be found in the [research paper][1] and
accompanying [github repo][2].

## Contributing
Code should be written for Python 3, include documentation (docstrings & comments), follow PEP 8 and pass all unittests.
To run the unittests, simply run `python -m unittest` from the repo directory.


[1]: http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
[2]: https://github.com/cjhutto/vaderSentiment