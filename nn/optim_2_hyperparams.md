# Hyperparameter Search



For a given machine learning model, there are usually numerous hyperparameters configuring it. Tweaking their values to reach the optimum performance of the model is what is referred to as __hyperparameter tuning__. This could be done manually, using ranges of possible values for each parameter and embedded `for` loops to go over all possible combinations. But it would be very tedious work and impractical as there may be too many possibilities - and recall that a single training involves numerous computations!

```{admonition} Exercise
:class: seealso
If your model has five hyperparameters and you want to try 10 different values for each of them, how many tuning combinations will there be?
```

Luckily, some tools are available to do the tuning!

## Grid Search
Here comes a great method called Grid Search. This process has been coded as a tool available in python libraries such as Scikit-Learn and PyTorch, as we will soon see. 

### Definition

````{prf:definition}
:label: gridsearchdef
__Grid Search__ is an exhaustive scan of the hyperparameters from manually specificed values in order to find the combination maximizing the model's performance.
````

The Grid Search method is also called _parameter sweep_ (although it should be called _hyperparameter sweep_). It is what it does: it sweeps over all possibilities of the hyperparameters the user provides. 

__How does Grid Search knows what is best?__  
The guidance here is a performance assessment, usually using a $k$-fold cross-validation. This is why most Grid Search modules in common libraries are named `GridSearchCV`, where the CV suffix stands for cross-validation. Which performance metric is used? This can be entered by the user, with of course proper thinking beforhand on which performance metric is the most relevant for the task at hand! The default in Scikit-Learn is the accuracy for classification and $r^2$ for regression. 

### GridSearchCV in Scikit-Learn
Below is an implementation of Grid Search in Scikit-Learn.

```python
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [3, 10, 30], 
               'max_features': [2, 4, 6, 8],
               'bootstrap': [True, False]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, 
                           param_grid, 
                           cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X, y)
```
<sub>From Aurélien Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ (Second Edition)</sub>

Grid Search requires at least two arguments: the model to optimize and the hyperparameter (search) space. In the code above, such a space is defined as a dictionary (`param_grid`), where the keys are the hyperparameter names and the dictionary values are lists of hyperparameter values that should be tested. 

````{tip}
It is recommended while deciding on which values to choose to first opt for consecutive powers of 10. In the case of a classification model involving a learning rate $\alpha$, a good call would be:

```python
param_grid = {
    "alpha": np.power(10, np.arange(-2, 1, dtype=float))
}
```
This would return `[0.01  0.1   1.]` for the values of $\alpha$ to test.
````

The results of the search are stored in the attribute `cv_results_`.

```python
cvres = grid_search.cv_results_
```
This function can print the `mean_test_score` along with the values for each hyperparameters:
```python
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

The best estimator is given by `grid_search.best_estimator_`. More details can be found on [Scikit-Learn Grid Search documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

### Limitations
Grid Search has several drawbacks:
* Its output is more a conditional "best" combination, as the search is limited to the hyperparameter values entered by the user. If the process returns the minimum or maximum value of a given hyperparameter range, it is likely the score will improve by extending the range. A bit of tweaking is still required from the user.
* Due to its exhaustivity, the search is time-consuming. All values must be assessed (and such $k$ times, with the cross-validation!) before ranking the combinations. It becomes impractical when the hyperparameter space is very large.

Here is when the Randomized Search comes into play.

## Randomized Search
### Definition
````{prf:definition}
:label: randomsearchdef
The __Randomized Search__ is a tuning method consisting of evaluating a given number of combinations by selecting them randomly.

The number of combinations is given by the user. 

````

It is a non-exhaustive search. Rather, a fixed number of parameter settings is sampled from the specified distributions. In Scikit-Learn it is the argument `n_iter`.

### RandomizedSearchCV in Scikit-Learn
Let's have a look at this example (source):

```python
import numpy as np

import scipy.stats as stats
from sklearn.utils.fixes import loguniform

from sklearn.model_selection import RandomizedSearchCV

# build a classifier
clf = SGDClassifier(loss="los_loss", penalty="elasticnet")

# specify parameters and distributions to sample from
param_dist = {
    "average": [True, False],
    "l1_ratio": stats.uniform(0, 1),
    "alpha": loguniform(1e-2, 1e0),
}

# run randomized search
n_iter_search = 15
random_search = RandomizedSearchCV(clf, 
                                   param_distributions=param_dist, 
                                   n_iter=n_iter_search
)

random_search.fit(X, y)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print( mean_score, params)

```

In the Randomized Search, the continuous hyperparameters should be set not as discrete values but as continuous distributions (it is highly recommended to do so to benefit from the randomization). The regulation hyperparameter `l1_ratio` above is defined as a uniform distribution from 0 to 1. The Randomized Search could pick any value from this interval. It is thus likely to be 'precise' than the Grid Search using a discrete set of values.

__The Pros__  
* The exploration is broad: the randomized search can discover non-intuitive combinations of hyperparameters, boosting the performance along the way (but the execution time is often longer).

* There is control from the user on the computing budget: it is restricted by the number of iterations. But it does not mean that the Randomized Search will take less time than the Grid Search; it all depends on the configuration.


## Summary

The exhaustive Grid Search method is good for a restricted hyperparameter space. It requires some prior knowledge from the users on ballparks and values of hyperparameters that are known to perform well (e.g. $\alpha$ between 0.01 and 0.1). It can be a first step for tuning hyperparameters.

The Randomized Search is preferable when the hyperparameter space is large. It can take longer but the user has more control on the execution time as it directly depends on the number of sampled combinations. It is the search to opt for when not knowing which hyperparameter values would work.

There are more advanced methods for hyperparameter tuning such as Bayesian Optimization and Evolutionary Optimization.


```{admonition} Learn More
:class: seealso

* Yoshua Bengio, [Practical Recommendations for Gradient-Based Training of Deep Architectures (2012)](https://arxiv.org/abs/1206.5533) (arXiv)  
* [Hyperparameter Optimization With Random Search and Grid Search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/) (machinelearningmastery.com)  
* [Comparing Randomized Search and Grid Search for Hyperparameter Estimation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html) (Scikit-Learn)  



```