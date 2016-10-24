# hyperparameters-optimization-talk


# About

This is a the accompanying repo for the Bdx hyperparameters optimization meetup.



# Installation

To install the project dependencies, run:

`pip install -r requirements`

If you are familiar with Conda, I would suggest creating a virtual environnement
and installing the dependencies in the following fashion:

`conda create --name hyperparameters-optimization --file requirements.txt`

Then activate the environment with the following command:

`source activate hyperparameters-optimization`


# Dataset

## Raw dataset

In this talk, we analyze an [IMDB](http://www.imdb.com/) dataset that you could find [here](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset).
Notice that the dataset is also available in the [data](/data) folder.

## Processed dataset

To get the processed dataset, run the following code
(don't forget to activate your virtual env):

```
python scripts/data_processing.py
```

# Slides

TODO: Add link

# Notebooks

TODO: Add link


# Resources


## Xgboost

* The [XGBoost](https://xgboost.readthedocs.io/en/latest/) website.

* The [XGBoost](https://github.com/dmlc/xgboost) Github page.

## Dataset

* You can find  [here](https://blog.nycdatascience.com/student-works/machine-learning/movie-rating-prediction/) a complete analysis from the dataset owner.  

## Hyperopt

* The [hyperopt](https://jaberg.github.io/hyperopt/) website.

* The [hyperopt](https://github.com/hyperopt/hyperopt) Github page.

# License

The MIT License (MIT)
