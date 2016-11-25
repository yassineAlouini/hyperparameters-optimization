# How to optimize hyperparameters?

At Qucit, we strive to improve the life of citizens by making cities more efficient and liveable. In order to make this dream a reality,
we are building a predictive platform.
At the core of this platform lies a set of general purpose machine learning algorithms that given enough urban data and a phenomenon (say vehicle congestion, parking fraud or shared car mobility), will predict it for future time values.

Moreover, the models

It is often hard and time-consuming to tune a machine learning and becomes infeasible when the model complexity …. How to tune it ? How to. First, let’s start with the basics of machine learning.


## A primer on machine learning

Machine learning is a systematic approch to "teach" computers how to automatically learn and predict from data without using hard-coded rules.

Let's say we want to predict the air quality. Given data about CO2, NOX, weather conditions and so on, a machine learning algorithm can do it.

## On the importance of hyperparameters

Wihtout tuning hyperparmeters, it is often easy to fall into the trap of overfitting.

To mitigate this issue, most of the time ML practionners will use cross-validation. It is a proxy that measures how well an algorithm will generalize by looking at a subset of a data.



## Strategies to optimize them

There are two broad families for optimizing hyperparameters:

* A methodic approach, called "grid search"
* A randomized approach

## Sequential model based optimization

This is a general approach.

## Hyperopt to the rescue

In order to find the best hyperparmeters, we will use [Hyperopt](https://github.com/hyperopt/hyperopt). As stated in the [website](http://hyperopt.github.io/hyperopt/):
> hyperopt is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

In order to find the best hyperparameters, one needs to specify two functions:

```Python
def score(params):
    params["gamma"] = np.log(params["gamma"])
    params["learning_rate"] = np.log(params["learning_rate"])
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    xgb_regressor = xgb.XGBRegressor(silent=False, **params)
    mse = cross_val_score(xgb_regressor,
                         train_features, train_targets,
                         cv=5, verbose=0, n_jobs=4,
                         scoring=mse_scorer).mean()
    rmse = np.sqrt(mse)
    return {'loss': rmse,
            'status': STATUS_OK}

def optimize(trials):
    space = hyperopt_cv_parameters
    best = fmin(score, space, algo=tpe.suggest,
                trials=trials,
                max_evals=MAX_EVALS)
    return best            

trials = Trials()
optimal_param = optimize(trials)
```


## Predicting the air quality in Paris (or another example)



Stay tuned for our upcoming blog posts. If you want to learn more about platform we are building, leave your email...
