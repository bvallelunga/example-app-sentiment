# Sentiment
Easily find the sentiment of a sentence.

**Possible Use Cases**
  * active support: route tweets and support emails to the proper support team.
     - negative sentiment messages to crisis support
     - neutral sentiment with automated replies
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
The output will map each input text to a score. Positive texts receive positive scores while negative texts receive negative
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
The underlying model was designed by a group at [Georgia Institute of Technology][1] and is "specifically attuned to 
sentiments expressed in social media".


## Want To Learn More?
See the [MAINTAINERS.md][2] to learn more about the underlying model as well as how to contribute to the app.


[1]: http://www.gatech.edu/
[2]: https://github.com/DopplerFoundation/example-app-sentiment/blob/master/MAINTAINERS.md
