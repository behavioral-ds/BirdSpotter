# BirdSpotter
BirdSpotter is a python package which provides an influence and bot detection toolkit for twitter. You should use this package if you have tweet dumps and you are interested in:
- The probability that a user is actually a bot.
- How influential a user is compared to their peers.

BirdSpotter is designed to be simple to use for researchers and people who have a limited coding experience.

## Set up
BirdSpotter is available though `pip` using `pip install birdspotter` within a terminal/`cmd`.

## Usage
BirdSpotter's primary function is to take a whole tweet dump and determine botness and influence metrics for each user. We expose this functionality through a class as follows.


```python
from BirdSpotter import BirdSpotter
bs = BirdSpotter()
```

We can extract the information from our dumps with `extractTweets()`


```python
bs.extractTweets('/path/to/tweet/dump.json')
```

To determine the 'botness' of the users in the dump we need to provide some bot training examples. If you don't have bot examples `train.pickle` is provided in the repository and can be downloaded. Alternatively, you can label your own data with the help of `getBotAnnotationTemplate()`


```python
bs.trainClassifierModel("train.pickle")
```

Finally, to get the individual users metrics we can run methods like `getBotness()`, which will return a pandas dataframe of each user with their 'botness' metric.


```python
bs.getBotness()
```

For more specific documentation refer to the [tutorial](./tutorial.ipynb) or the [documentation](http://birdspotter.rtfd.io/).

## References
- Rizoiu, M.A., Graham, T., Zhang, R., Zhang, Y., Ackland, R. and Xie, L., 2018, June. # DebateNight: The Role and Influence of Socialbots on Twitter During the 1st 2016 US Presidential Debate. In Twelfth International AAAI Conference on Web and Social Media.

## Acknowledgements
This project is funded through the UTS Data Science Institute.


## License
Both dataset and code are distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license, a copy of which can be obtained following this [link](https://creativecommons.org/licenses/by-nc/4.0/). If you require a different license, please contact us at Marian-Andrei@rizoiu.eu
