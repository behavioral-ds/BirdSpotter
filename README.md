<img src="https://raw.githubusercontent.com/behavioral-ds/BirdSpotter/master/birdspotter_logo.png" alt="logo" width="150"/> 

# `birdspotter`: A tool to measure social attributes of Twitter users

[![PyPI status](https://img.shields.io/pypi/status/birdspotter.svg)](https://pypi.python.org/pypi/birdspotter/) [![PyPI version fury.io](https://badge.fury.io/py/birdspotter.svg)](https://pypi.python.org/pypi/birdspotter/) [![Documentation Status](https://readthedocs.org/projects/birdspotter/badge/?version=latest)](http://birdspotter.readthedocs.io/?badge=latest)

`birdspotter` is a python package providing a toolkit to measures the _social influence_ and _botness_ of twitter users. It takes a twitter dump input in `json` or `jsonl` format and produces measures for:
- **Social Influence**: The relative amount that one user can cause another user to adopt a behaviour, such as retweeting.
- **Botness**: The amount that a user appears automated.

## References:
```
Rizoiu, M.A., Graham, T., Zhang, R., Zhang, Y., Ackland, R. and Xie, L. # DebateNight: The Role and Influence of Socialbots on Twitter During the 1st 2016 US Presidential Debate. In Twelfth International AAAI Conference on Web and Social Media (ICWSM'18), 2018. https://arxiv.org/abs/1802.09808
```
```
Ram, R., & Rizoiu, M.-A. A social science-grounded approach for quantifying online social influence. In Australian Social Network Analysis Conference (ASNAC'19) (p. 2). Adelaide, Australia, 2019.
```

## Installation
`pip3 install birdspotter`
##### `birdspotter` requires a python version `>=3`.

## Basic Usage
##### To use `birdspotter` on your own twitter dump, replace './tweets.20150430-223406.jsonl' with the path to your twitter dump './path/to/tweet/dump.json'. In this example we use the nltk twitter-sample dataset found on kaggle. It can be downloaded [here](https://www.kaggle.com/nltkdata/twitter-sample#tweets.20150430-223406.json). Notably the file extension needs to be changes from `.json` to `.jsonl`.

 ```python
from birdspotter import BirdSpotter
bs = BirdSpotter('./tweets.20150430-223406.jsonl')
# This may take a few minutes, go grab a coffee...
labeledUsers = bs.getLabeledUsers(out='./output.csv')
```

After extracting the tweets, `getLabeledDataFrame()` returns a `pandas` dataframe with the influence and botness labels of users and writes a `csv` file if a path is specified i.e. `./output.csv`.

 ##### `birdspotter` relies on the [Fasttext word embeddings](https://fasttext.cc/docs/en/english-vectors.html) [wiki-news-300d-1M.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip), which will automatically be downloaded if not available in the current directory (`./`) or a relative data folder (`./data/`).

### Get Cascades Data
After extracting the tweets, the retweet cascades are accessible by using:
```python
cascades = bs.getCascadesDataFrame()
```
##### This dataframe includes the expected structure of the retweet cascade as given by Rizoiu et al. (2018) via the column `expected_parent` in this dataframe.

## Analysis



## How to train the classifier with your own botness data
`birdspotter` provides functionality for training the botness detector with your own training data. After extracting the tweets, we run:
```python
bs.getBotAnnotationTemplate('./annotation_file.csv')
```
This produces a `csv`, with an empty column `isbot` to be annotated by a human, as below:

|  | screen_name   | user_id    | isbot | 
|-------|---------------|------------|-------| 
| 0     | 007_Rebooted  | 232358211  |       | 
| 1     | 0151Sam64     | 351532518  |       | 
| 2     | 0192am        | 621237594  |       | 
| 3     | 01EddyCordero | 324670614  |       | 
| 4     | 052Erik       | 2807483094 |       | 
| ...   | ...           | ...        | ...   | 
| 11719 | zoommonk      | 304081311  |       | 
| 11720 | zosephh       | 453328842  |       | 
| 11721 | zoumrouda     | 384483993  |       | 
| 11722 | zwartekat     | 14442577   |       | 
| 11723 | zygoticdeb    | 21486509   |       | 


Once annotated the botness detector can be trained with:
```python
bs.trainClassifierModel('./annotation_file.csv')
```
Finally, to get the new botness scores we run:
```python
bs.getBotness()
```
## Advanced Usage
<!-- ### Adding more influence metrics
`birdspotter` provides DebateNight influence as a standard, when `getLabeledUsers` is run. To generate spatial-decay influence run:
```python
bs.getInfluenceScores(time_decay = -0.000068, alpha = 0.15, beta = 1.0)
```
This returns the updated `featureDataframe` with influence scores appended, under the column `influence (<alpha>,<time_decay>,<beta>)`. -->

### Defining your own word embeddings
`birdspotter` provides functionality for defining your own word embeddings. For example:
```python
customEmbedding # A mapping such as a dict() representing word embeddings
bs = BirdSpotter('./tweets.20150430-223406.jsonl', embeddings=customEmbedding)
```

Embeddings can be set through several methods, refer to [setWord2VecEmbeddings]().

##### Note the default bot training data uses the [wiki-news-300d-1M.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) and as such we would need to retrain the bot detector for alternative word embeddings.
## Alternatives to python
### Command-line usage
 `birdspotter` can be accessed through the command-line to return a `csv`, with the recipe below:

 ```
birdspotter ./path/to/twitter/dump.json ./path/to/output/directory/
 ```
### R usage
`birdspotter` functionality can be accessed in `R` via the [`reticulate`](https://rstudio.github.io/reticulate/) package. `reticulate` still requires a `python` installation on your system and `birdspotter` to be installed. The following produces the same results as the standard usage.

```r
install.packages("reticulate")
library(reticulate)
use_python(Sys.which("python3"))
birdspotter <- import("birdspotter")
bs <- birdspotter$BirdSpotter("./tweets.20150430-223406.jsonl")
bs$getLabeledDataFrame(out = './output.csv')
```

## Acknowledgements
The development of this package was partially supported through a UTS Data Science Institute seed grant.
