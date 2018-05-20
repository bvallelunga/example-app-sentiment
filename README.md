# Sentiment
Easily find the sentiment of a sentence targeted for social media and support messages.

**Possible Use Cases**
  * Support Routing: route inbound messages to the proper support team.
    - Have a dedicated crisis response team for negative messages
    - Use chat bots or automated replies for neutral messages
  * Discover extremely positive messages as showcases of happy customers
  * Filter extremely negative messages in chat rooms
  

## Input Scheme
The input should contain an array of texts. 
``` json
{
  "texts": ["What a beautiful product.", "The color is very dull."]
}
```

## Output Scheme
The output will map each input message to a score. Positive texts receive positive scores while negative texts receive negative
scores. 
 
``` json
{
  "texts": 
    {
      "The book was good.": 0.4404,
      "At least it isn't a horrible book.": 0.431,
      "The book was only kind of good.": 0.3832,
      "Today SUX!": -0.5461,
      "John is not smart, handsome, nor funny.": -0.7424,
      "The plot was good, but the characters are uncompelling and the dialog is not great.": -0.7042
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
