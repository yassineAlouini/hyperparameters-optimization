# How we optimize hyperparameters at Qucit?


## Summary

### en
Finding the optimal hyperparameters is often tricky and time-consuming. Common methods consist of testing all the possible combinations or drawing them at random from some distributions. What if there is a better way?
Find out by reading our latest blog post, where we explain how we use hyperopt to fine-tune machine learning algorithms used at Qucit.


### fr
La recherche d'hyperparmètres optimaux est souvent délicate et laborieuse.
Généralement, nous utilisons une recherche exhaustive de toutes les combinaisons possibles ou un tirage aléatiore. Et s'il existait une meilleure approche ?

Découverez comment, chez Qucit, nous utilisons hyperopt pour calibrer nos modèles de machine learning.


## Introduction

At Qucit, we strive to improve the life of citizens by making cities more efficient and enjoyable. <br> In order to make this dream a reality, we are building a predictive platform. <br>
At the core of this platform lies a set of general purpose machine learning algorithms that, given enough urban data and a phenomenon (say vehicle congestion, parking fraud or shared car mobility), will predict it for future time values.

These various models need fine-tuning to work properly. The tuning step is
often laborious, time-consuming and intractable when the model becomes very
complex.  

What are the strategies that we employ at Qucit to solve this problem?
Let’s start from the beginning.


## A primer on machine learning

Machine learning is a systematic approach to "teach" computers how to
automatically learn and extract "insights" from data without using hard-coded rules
(in opposition with what was previously done in the [expert system](https://en.wikipedia.org/wiki/Expert_system) method).

## Model complexity and the curse of overfitting

<img src='http://1.bp.blogspot.com/-CQi8z9YYDzI/T9WYh8hdhQI/AAAAAAAAAv8/Mf8E9fIwIps/s1600/p1.png'>

Often, we want our models to perform well. An intuitive method to improve the
performances is by making the models more and more complex by adding lots of hyperparameters.

However this approach doesn't work. In fact, without tuning hyperparameters, it is often easy to fall into the trap of overfitting (which lies in the right side of the graph above). This happens when the algorithm fits too much to the training data and thus is poorly suited to generalize.

So how to avoid this curse while getting good performance?


## Strategies to select hyperparameters

Ideally, we would like to select hyperparameters that give us the best training performance while being able to generalize. In order to do that, ML practionners often use cross-validation evaluation. So how does this technique work?

### Cross-validation

<img src='http://vinhkhuc.github.io/assets/2015-03-01-cross-validation/5-fold-cv.png'>

Cross-validation is one of the most popular methods to **estimate** how well an ML model can **generalize** on unseen data. It consists in dividing the training data into n-folds (generally 5 folds as in the graph above). Then using all but one fold we train the model and evaluate its performances on the emaining one. This step is repeated n times by varying the fold that is used for evaluation. The final score is the average of the n scores obtained on each iteration.

Applied to the context of hyperparameters, it gives us a method to estimate how well each different model (that differs by its hyperparameters' values) generalizes and thus select the best one.

Now that we have our evaluation method, one question remains to be solved: how to choose the hyperparameters grid over which to optimize? Let's find out.

### Choosing the grid points

There are two broad families for choosing the hyperparameters values to test:

* A deterministic and exhaustive approach, known as grid search, which consists in trying all the possible combinations
* A randomized approach where grid points are drawn from specified distributions.

There is also a third, more clever method that leverages [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics).

## Sequential Model-based Global optimization (SMBO)

This is a general optimization method that consists in approximating a costly evaluation function by a more tractable *proxy*. One commonly used approximation method for SMBO is the Tree-structured Parzen Estimators algorithm (TPE for short). It uses Parzen estimators (also known as [kernel density estimators](https://en.wikipedia.org/wiki/Kernel_density_estimation)) to replace a complex prior with estimated non-parametric densities. For more technical details, we refer to this [paper](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf).

## Hyperopt to the rescue

[Hyperopt](https://github.com/hyperopt/hyperopt) is an open-source Python library that implements the TPE approach.

As stated in its [website](http://hyperopt.github.io/hyperopt/):
> hyperopt is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

Now, in order to find the best hyperparameters using hyperopt, one needs to specify two functions - `loss` and `optimize`, a hyperparameters grid and a machine learning model (an [xgboost](https://xgboost.readthedocs.io/en/latest/) regression model in this example).

On one hand, the `loss` function gives the evaluation metric that is used to find the best hyperparameters. It is computed using cross validation for an XGboost regression model.
On the other hand, the `optimize` function specifies the optimization strategy and the search space.

Finally, notice that we have introduced a `transform_params` function since Hyperopt doesn't return the hyperparameters values as expected (`exp(value)` instead of `value` when working with a logarithmic distribution for example).

Here is the code in Python:

```Python

# Machine learning imports
import xgboost as xgb
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.datasets import make_regression


# Hyperopt imports
from hyperopt import fmin, tpe, STATUS_OK, Trials


MAX_EVALS = 100
SEED = 314 # To make the example reproducible
NUMBER_SAMPLES = 2000

hyperopt_hp_grid = {'n_estimators': hp.quniform('n_estimators', 10, 1000, 1),
'learning_rate' : hp.loguniform('learning_rate', 0.001, 0.1),
'max_depth' : hp.quniform('max_depth', 3, 15, 1),
'gamma': hp.loguniform('gamma', 0.01, 1)}

mse_scorer = make_scorer(mean_squared_error)

# Generate fake regression data
features, targets = make_regression(NUMBER_SAMPLES, random_state=SEED)
# Train/test split
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=TEST_SIZE, random_state=SEED)

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

## Predicting car accidents' casualties

Now that the we have discovered how to optimize hyperparameters using Hyperopt, let's apply this knowledge to a concrete use case.

We will explore the 2015 USA car accidents data set.

You can get the `accidents.csv` dataset from [Kaggle](https://www.kaggle.com/nhtsa/2015-traffic-fatalities) (you will need to create an account first) or from the following [Github repo](https://github.com/yassineAlouini/hyperparameters-optimization-talk/tree/master/data). For further information about the dataset variables, you should check this exhaustive [report](https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315).

To follow along, it is highly recommended to get the accompanying  [notebook](https://github.com/yassineAlouini/hyperparameters-optimization-talk/notebooks/how_we_optimize_hyperparameters_at_qucit.ipynb).

First, let 's load the data:

```Python
def load_car_accidents():
    accidents_df = pd.read_csv('../data/accident.csv')
    features = accidents_df.drop('FATALS', axis=1)
    # Don't select the columns having object type
    columns = features.columns[features.dtypes != 'object']
    targets = accidents_df[['FATALS']]
    return features.loc[:, columns], targets

features, targets = load_car_accidents()
```

The targets column we will use is the `FATALS` one. It contains the number of fatal
casualties for each accident. As you might have noticed, we have dropped
all the columns with `object` type. This is done in order to avoid processing
the data.

Once we have loaded the data, we split it into train and test subsets (we leave the test subset for the final evaluation).


```Python

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=TEST_SIZE, random_state=SEED)
```

As discussed in the previous parts, we will use the following three algorithms:
grid search, random search and TPE.

### Results

Running the three optimization strategies gives the following results:

<table border="1" class="dataframe">
		<thead>
			<tr style="text-align: right;">
				<th></th>
				<th>optimal_hyperparameters</th>
				<th>test_rmse</th>
				<th>mean_cv_rmse</th>
				<th>std_cv_rmse</th>
				<th>train_rmse</th>
				<th>opt_method</th>
			</tr>
		</thead>
		<tbody>
			<tr>
				<th>0</th>
				<td>{u\'n_estimators\': 25, u\'learning_rate\': 0.8, u\'max_depth\': 10, u\'gamma\': 0.2}</td>
				<td>0.390102</td>
				<td>0.406295</td>
				<td>0.001396</td>
				<td>0.138684</td>
				<td>grid_search</td>
			</tr>
			<tr>
				<th>1</th>
				<td>{u\'n_estimators\': 105, u\'learning_rate\': 0.0583873644628, u\'max_depth\': 5, u\'gamma\': 0.555704565...</td>
				<td>0.325454</td>
				<td>0.338373</td>
				<td>0.003867</td>
				<td>0.302278</td>
				<td>random_search</td>
			</tr>
			<tr>
				<th>2</th>
				<td>{u\'n_estimators\': 721, u\'learning_rate\': 0.0812468270757, u\'max_depth\': 4, u\'gamma\': 0.640681919...</td>
				<td>0.325325</td>
				<td>0.338179</td>
				<td>0.003728</td>
				<td>0.298393</td>
				<td>hyperopt_tpe</td>
			</tr>
		</tbody>
</table>

The TPE algorithm gives the best results on both test and CV sets (the lower is the loss, the better it is) followed closely by the random search. The grid search approach is the worst as expected. We have also included the train losses to show that the grid search method is the one that yields the largest overfitting (measured as the difference between test and train losses).

## Conclusion

The approach that we have presented in this blog post seems like magic.
It really isn't.

In fact, we had to provide the grid search hyperparameters distributions (the type of the distribution and its support).

Moreover, it can't be as easily distributed as grid search even though is is possible to do it [using MongoDB](https://github.com/hyperopt/hyperopt/wiki/Parallelizing-Evaluations-During-Search-via-MongoDB).

Finally, notice that there is another popular alternative to the TPE algorithm that uses [Gaussian processes] (https://en.wikipedia.org/wiki/Gaussian_process)(a generalization of the Gaussian distribution).   [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) is one such implementation in Python.

If you want to learn more about the subject of hyperparameters tuning, I highly recommend reading this [blog post](http://sebastianraschka.com/blog/2016/model-evaluation-selection-part3.html).

Finally for a more exhaustive theoretical study, check the following [paper](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf).

Stay tuned for our upcoming blog posts. If you want to have a sneak peek at our platform when it is ready, please leave your email here. And we promise, we won't fill your email box with spam.
