## Stock market prediction from news headlines

Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved [2019-08-19] from https://www.kaggle.com/aaron7sun/stocknews

## Target

The target for this task is

## Metric

The metric we are using is AUCROC (often called AUC) - area under the reciever operating characteristic curve.  We will use the `sklearn` implementation:

```python
from sklearn.metrics import roc_auc_score

metric = roc_auc_score(y_true, y_score)
```

## Data

Your training data is from 2008-06-08 to 2014-11-20.  Your test period is from 2014-11-21 to 2016-07-01.

This dataset was taken from the Kaggle competition [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews/), which was designed for an NLP course.

The dataset is from two sources 
- the r/worldnews Reddit - the top 25 headlines ranked by upvotes
- the Dow Jones Industrial Average (DJIA)

The data is supplied as three csvs.  Download the data using

```bash
unzip data/stocknews.zip -d data
```

You can then generate the train and test data using

```bash
python data.py --test 0

I have randomly dropped data at a rate of 5% for all the csvs.

