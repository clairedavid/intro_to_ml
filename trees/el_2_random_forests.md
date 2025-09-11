# Random Forests

## Presentation

````{prf:definition}
:label: randomforestdef

A __random forest__ is an ensemble learning method made of solely decision trees.  

The predicted class is the one obtaining the majority of votes across all decision trees in the forest.
````

Random forest classifiers are implemented in Scikit-Learn from the `ensemble` library. Here is a minimal code:

```python

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
```

Only four lines. But there are obviously much more lines of code 'backstage' behind those calls. It's important to understand what these methods do as well as their arguments.
* `n_estimators` refers to the number of trees in the forest
* `max_leaf_nodes` is a hyperparameter regularizing each tree 
* `n_jobs` tells how many CPU cores to use (-1 tells Scikit-Learn to use all cores available)

The Random Forest inherits all hyperparameters of the decision trees. There are other hyperparameters proper to Random Forest classifiers, for instance activating the bagging `bootstrap=True` with `max_samples` controlling the sample size of each subset.  
More on [Scikit-Learn RandomForest page](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).


## Advantages

Random forests are popular as they present numerous qualities:
* their results are among the most accurate compared with other machine learning algorithms
* they are less likely to overfit the data
* they run quickly, even with large datasets
* they require less data to train
* they are easy to interpret
* they can be scaled using parallel computing (multiple cores)
* they can rank features by their relative importance

More about the last point: random forest classifiers not only classify the data, they can also measure the relative effect, or importance, each feature has with respect to the others. This is very handy while designing a machine learning algorithm to see which features matter. If for instance there are too many features, a random forest can help reduce this number while not removing key features. Scikit-Learn computes the feature importance automatically after training. It is expressed as a score, ranging from zero (useless feature) to 1 (most important). Here is a snippet showing how to access the feature importance scores:

```python
# Dataset loading (not written): X and y dataframes

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(X, y)

for name, score in zip(X.columns, rnd_clf.feature_importances_)
    print(name, score)
```

This will output something in the like:
```
    x1  0.11323
    x2  0.54530
    x3  0.05961
    ...
```
The sum of all feature importance is 1.

We have seen the advantage of decision trees working together in a beautiful forest to produce low variance predictions. Let's go one step further: the boosting.


