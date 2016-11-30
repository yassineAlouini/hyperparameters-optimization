

# How to optimize hyperparameters?


## Summary

### en
Finding the optimal hyperparameters is often tricky and time-consuming. Common methods consist of testing all the possible combinations or drawing them at random from some distributions. What if there is a better way?
Find out by reading our latest blog post, where we explain how we use hyperopt to fine-tune our machine learning algorithms.


### fr
La recherche d'hyperparmètres optimaux est souvent une tâche délicate et laborieuse. Pour y arriver, une recherche exhaustive de toutes les combinaisons possible ou bien un tirage aléatiore sont employés. Existe-t-il une meilleure approche ?

Découverez dans notre dernier poste de blog comment nous y parvenons en employant hyperopt pour calibrer nos modèles de machine learning.

Finding the optimal hyperparameters is often tricky and time-consuming. Common methods consist of testing all the possible combinations or drawing them at random. What if there is a better way?
Find out by reading our latest blog post, where we explain how we use hyperopt at Qucit to fine-tune our machine learning algorithms.            

## Introduction

At Qucit, we strive to improve the life of citizens by making cities more efficient and liveable. <br> In order to make this dream a reality, we are building a predictive platform. <br>
At the core of this platform lies a set of general purpose machine learning algorithms that, given enough urban data and a phenomenon (say vehicle congestion, parking fraud or shared car mobility), will predict it for future time values.

These various models need fine-tuning to work properly. The tuning step is
often laborious, time-consuming and intractable when the model is complex.  

What are the strategies that we employ at Qucit to solve this problem ?
Let’s start from the beginning.


## A primer on machine learning

Machine learning is a systematic approach to "teach" computers how to
automatically learn and extract "insights" from data without using hard-coded rules
(in opposition with what was previously done in the [expert system](https://en.wikipedia.org/wiki/Expert_system) method).

Let's say we want to predict the air quality in a city. Given data about CO2, NOx, weather conditions and so on, a machine learning algorithm can do it.

## Model complexity and the curse of overfitting


<img src='http://1.bp.blogspot.com/-CQi8z9YYDzI/T9WYh8hdhQI/AAAAAAAAAv8/Mf8E9fIwIps/s1600/p1.png'>

Often, we want our models to perform well.

Without tuning hyperparameters, it is often easy to fall into the trap of overfitting.

To mitigate this issue, most of the time ML practionners will use cross-validation. It is a proxy that measures how well an algorithm will generalize by looking at a subset of a data.

<Wiki >
 Hyperparameter optimization contrasts with actual learning problems, which are also often cast as optimization problems, but optimize a loss function on the training set alone. In effect, learning algorithms learn parameters that model/reconstruct their inputs well, while hyperparameter optimization is to ensure the model does not overfit its data by tuning, e.g., regularization.
</Wiki>

## Strategies to select hyperparameters

Most of the times, ML practionners will use cross-validation evaluation to select hyperparameters. So what is this technique?

### Cross-validation

<img src='http://vinhkhuc.github.io/assets/2015-03-01-cross-validation/5-fold-cv.png'>


Cross-validation is a popular method to estimate how well a ML model can generalize. It consists in ...

In the context of hyperparameters optimization it consists in selecting the best performing ones on a subset of the training data. The diagram above shows a 5-fold cross-validation.

Now that we have our evaluation method, we need to specify the grid values. 

### Choosing the grid points

There are two broad families for choosing the hyperparameters values to test:

* A deterministic and exhaustive approach, titled grid search, which consists in trying all the possible grid points combinations
* A randomized approach where grid points are drawn from specified distributions.

There is also a third, more clever method that leverages Bayesian statistics.


## Sequential Model-based Global optimization

This is a general approach .

## Hyperopt to the rescue

In order to find the best hyperparmeters, we will use [Hyperopt](https://github.com/hyperopt/hyperopt).

As quoted in the [website](http://hyperopt.github.io/hyperopt/):
> hyperopt is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

In order to find the best hyperparameters using hyperopt, one needs to specify two functions `loss` and `optimize`, a hyperparameters grid and a machine learning model (an [xgboost](https://xgboost.readthedocs.io/en/latest/) regression model in this example).
Here is the code in Python:

```Python

# Machine learning imports
import xgboost as xgb
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# Hyperopt imports
from hyperopt import fmin, tpe, STATUS_OK, Trials


MAX_EVALS = 100
mse_scorer = make_scorer(mean_squared_error)

hyperopt_hp_grid = {'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
'learning_rate' : hp.loguniform('learning_rate', 0.001, 0.1),
'max_depth' : hp.quniform('max_depth', 3, 15, 1),
'gamma': hp.loguniform('gamma', 0.01, 1)}

def transform_params(params):
    params["gamma"] = np.log(params["gamma"])
    params["learning_rate"] = np.log(params["learning_rate"])
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    return params


def loss(params):
   params = transform_params(params)
   xgb_regressor = xgb.XGBRegressor(silent=False, **params)
   cv_mse = cross_val_score(xgb_regressor, train_features, train_targets,
                         cv=5, verbose=0, n_jobs=4,
                         scoring=mse_scorer)
   rmse = np.sqrt(cv_mse.mean())
   return {'loss': rmse,
           'status': STATUS_OK}

def optimize(trials, space):
    best = fmin(loss, space, algo=tpe.suggest,
                trials=trials,
                max_evals=MAX_EVALS)
    return best            

trials = Trials()
hyperopt_optimal_hp = optimize(trials, hyperopt_hp_grid)
hyperopt_optimal_hp = transform_params(hyperopt_optimal_hp)
```


## Predicting car accidents

Now that the we have learned how to optimize hyperparameters using Hyperopt, let's apply this knowledge to a concrete case.

You can find the accompanying notebook here.


## Conclusion

The approach that we have presented in this blog post seems like magic. It really isn't.

In fact, we had to provide the grid search hyperparameters distributions (the type of the distribution and its support).

Moreover, it can't be as easily distributed as grid search even though is is possible to do it (using MongoDB)[https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB].

Finally, notice that there is another popular alternative to the TPE algorithm that uses [Gaussian processes](https://en.wikipedia.org/wiki/Gaussian_process)(a generalization of the Gaussian distribution).   [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) is one such implementation in Python.

If you want to learn more about the subject of hyperparameters tuning, I would highly recommend reading this [blog post](http://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html).

Finally for a more exhaustive theoretical study, check the following [paper](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf).

Stay tuned for our upcoming blog posts. If you want to have a sneak peek at our platform when it is ready, please leave your email here. And we promise, we won't fill your email box with spam.

## References

* https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
* https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
