# hyperparameters-optimization


# About

This is the accompanying repo for the Bdx hyperparameters optimization meetup.
It also contains the notebook and dataset for [this](http://www.qucit.com/2016/12/06/how-to-optimize-hyperparameters/) Qucit blog post
and also the hyperparameters optimization webinar.


# Installation

To install the project dependencies, run:

`pip install -r requirements`

If you are familiar with Conda, I would suggest creating a virtual environnement
and installing the dependencies in the following fashion:

`conda create --name hyperparameters-optimization --file requirements.txt`

Then activate the environment with the following command:

`source activate hyperparameters-optimization`


# Datasets

## IMDB

### Raw dataset

In this talk, we analyze an [IMDB](http://www.imdb.com/) dataset that you could find [here](https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset).
Notice that the dataset is also available in the [data](/data) folder (titled "movies_metadata.csv").

### Processed dataset

To get the processed dataset, run the following code
(don't forget to activate your virtual env):

```
python scripts/imdb_data_processing.py
```

## Airlines Delay

### Raw dataset

In the blog post, we analyze the Airlines Delay dataset that you could find [here](https://www.kaggle.com/giovamata/airlinedelaycauses).
Notice that the dataset is also available in the [data](/data) folder
(titled "DelayedFlights.csv.zip").

## Results

The live demo results are stored [here](/data/hyperparameters_selection_results.csv).

# Slides

The talk slides are available [here](talk_slides.pdf). <br>
The webinar slides are available [here](webinar_slides.pdf).

# Notebooks

You can check the different notebooks used during the talk and the live demo by browsing the [notebooks](/notebooks) folder. It also contains the accompanying notebook for the blog post.
The `/notebooks/webinar` [folder](/notebooks/webinar) contains the webinar notebook.

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
