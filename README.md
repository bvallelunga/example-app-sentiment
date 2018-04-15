# Sentiment
Easily find the sentiment of a sentence; trained on millions of praising, neutral, and negative Amazon reviews.

**Possible Use Cases**
  * active support: route tweets and support emails to the proper support team.
     - negative sentiment messages to crisis support
     - neutral sentiment with automated replys
  * discover extremely positive messages as showcases of happy customers
  * filter extremely negative messages in chat rooms
  

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
