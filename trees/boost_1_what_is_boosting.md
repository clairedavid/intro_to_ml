# What is Boosting?

The general idea behind boosting is a correction of previous learners by the next row of classifiers.

````{prf:definition}
:label: boostingdef
__Boosting__ refers to ensemble methods that are tuning weak learners into a strong one, usually sequentially, with the next predictor correcting its predecessors.
````
Usually the predictors are really shallow trees, namely one root nodes and two final leaves, aka a decision stump (see {prf:ref}`decisionstumpdef`). 


Many boosting methods are available. We will see two popular ones: AdaBoost and Gradient Boosting.