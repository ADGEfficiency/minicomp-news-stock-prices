# Stock market prediction from news headlines

`Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved 2019-08-19 from https://www.kaggle.com/aaron7sun/stocknews`

## Target

The task is to predict the daily change in the Dow Jones Industrial Average (DJIA).  The task is formulated as a binary classification task:
- `0` when DJIA Adj Close value decreased
- `1` when DJIA Adj Close value rose or stayed as the same

## Metric

The metric is AUCROC (often called AUC) - area under the receiver operating characteristic curve.  We will use the `sklearn` implementation.  Note that `y_score` is the probability estimate of the positive class:

```python
from sklearn.metrics import roc_auc_score

metric = roc_auc_score(y_true, y_score)
```

## Data

This dataset was taken from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/).

The dataset is from two sources:
- the r/worldnews Reddit - the top 25 headlines ranked by upvotes
- Yahoo Finance

The data is supplied as three csvs.  The raw data is in `./data/stocknews.zip`.  Run the commands below to unzip it into the correct place, and to generate the training set (you will need `pandas` and `numpy` to do this):

```bash
unzip data/stocknews.zip -d data
python data.py
rm -rf data/raw
```

You should only ever use the data in `./data/train` for training.

The data is split:
- training - 2008-06-08 to 2014-11-20
- test - 2014-12-03 to 2016-07-01

### Test time data

You will only get `RedditNews` and `Combined_News_DJIA_train`.  Generate the data using the command below - run only a test time!

```python
python data.py --test 1
```

## Potential Approaches

There are a few approaches to take here - it is likely that some combination will work well.

An NLP approach - use the headlines to generate features (word counts etc):
- data cleaning (such as `.lower()`)
- a good starting point is a bag of words approach (use the `CountVectorizer` and/or `TfidfVectorizer` from `sklearn`)
- you can then move to an `n-gram` approach (looking at groups on `n` words)
- a more complex approach might involve word embeddings

A time series approach - use previous values of the series + other series:
- think about how many previous values you will have at test time
- taking the log is useful to turn a multiplicative process (such as accumulating stock prices) into an additive process
- standardization/normalization
 
A time feature approach 
- use datetime features (day of week etc)
