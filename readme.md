## Stock market prediction from news headlines

`Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved [2019-08-19] from https://www.kaggle.com/aaron7sun/stocknews`

## Target

This is a binary classification task:
- $1$ when DJIA Adj Close value rose or stayed as the same
- $0$ when DJIA Adj Close value decreased

## Metric

The metric we are using is AUCROC (often called AUC) - area under the receiver operating characteristic curve.  We will use the `sklearn` implementation:

```python
from sklearn.metrics import roc_auc_score

metric = roc_auc_score(y_true, y_score)
```

## Data

Your training data is from 2008-06-08 to 2014-11-20.  Your test period is from 2014-12-03 to 2016-07-01.  The test data is the same as the train except the two `label` columns are removed.

This dataset was taken from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/), which was designed for an NLP course.

The dataset is from two sources 
- the r/worldnews Reddit - the top 25 headlines ranked by upvotes
- the Dow Jones Industrial Average (DJIA)

The data is supplied as three csvs.  The raw data is in `./data/stocknews.zip`.  Run the commands below to unzip it into the correct place, and to generate the training set (you will need `pandas` and `numpy` to do this):

```bash
unzip data/stocknews.zip -d data
python data.py
```

This dataset is unclean & untrusted - be cautious!  Some data is duplicated across the three csvs.  I have randomly dropped data at a rate of 5% for all the csvs.

## Approaches

There are two clear approaches to take here:
1. use the headlines
2. use previous values of the series

If you are taking the second approach, think about what data you will have available at test time.
