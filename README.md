# Sentiment
Discover the sentiment of texts. Trained on millions of Amazon reviews."

**Possible Use Cases**
  * score the tone of an email in an email client
  * understand the general perception of a product


## Input Scheme
The input should contain an array of texts. 
``` json
{
  "texts": ["What a beautiful product.", "The color is very dull."]
}
```

## Output Scheme
The output will map each input text to a score. Positive receive positive scores while negative texts receive negative 
scores. 
 
``` json
{
  "texts": 
    {
      "What a beautiful product.": {"score": 1.15},
      "The color is very dull.": {"score": -0.88}
    }
 }
```


## Training
The model was trained by a [group][1] at [OpenAI][2] on a dataset of 82 million Amazon product reviews.


[1]: https://arxiv.org/pdf/1704.01444.pdf
[2]: https://openai.com/
