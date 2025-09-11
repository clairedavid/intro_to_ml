# AdaBoost

AdaBoost is short for Adaptative Boosting. It works by assigning larger weights to data samples misclassified by the previous learner. Let's see how this work.

## Algorithm

### Pseudocode

````{prf:algorithm} AdaBoost
:label: AdaBoostalgo


__Inputs__  
Training data set $X$ of $m$ samples

__Outputs__  
A collection of decision boundaries segmenting the $k$ feature phase space.

__Initialization__  
Each training instance $x^{(i)}$ is given the same weight 
\begin{equation*}
w^{(i)} = \frac{1}{m}
\end{equation*}

__Start__  
__For__ each predictor $t = 1 , \dots , N^\text{pred}$  
  a. Train on all samples and compute the weighted error rate $r_t$ 
   \begin{equation}
        r_t = \frac{\sum_{i = 1}^m w^{(i)} [ \hat{y}_t^{(i)} \neq y^{(i)} ]}{\sum_{i = 1}^m w^{(i)}}
   \end{equation}
  b. Give the predictor $t$ a weight $W_t$ measuring accuracy  
   \begin{equation}
        W_t = \alpha \log \frac{1 - r_t}{r_t}
   \end{equation}
   $W_t$ points to a high number if the predictor is good, zero if the predictor is guessing randomly, negative if it is bad. 
   $\alpha$ is the learning rate.
   
  c. Update the weights of all data samples:
   \begin{equation}
        w^{(i)} \longleftarrow
        \begin{cases} 
        w^{(i)} & \text{if } \hat{y}_t^{(i)} = y^{(i)} \\[1ex]
        w^{(i)} \, \exp(W_t) & \text{if } \hat{y}_t^{(i)} \neq y^{(i)}
        \end{cases}
   \end{equation}
  d. Normalize the weights
   \begin{equation}
    w^{(i)} \rightarrow \frac{w^{(i)}}{\sum_{i = 1}^m w^{(i)}}
   \end{equation}

__Exit conditions__  
* $N^\text{pred}$ is reached
* All data sample are correctly classified (perfect classifier)

````

### A visual

The illustration below gives a visual of the algorithm.

```{figure} ../images/boost_2_adaboost.png
---
  name: boost_2_adaboost
  width: 80%
---
 . Visual of AdaBoost.  
 Misclassified samples are given a higher weight for the next predictor.  
 Base classifiers are decision stumps (one-level tree).  
 <sub>Source: modified work by the author, originally from subscription.packtpub.com</sub>
 ```

```{note}
As the next predictor needs input from the previous one, the boosting is not an algorithm that can be parallelized on several cores but demands to be run in series.
```


How does the algorithm make predictions? In other words, how are all the decision boundaries (cuts) combined into the final boosted learner?  

### Combining the predictors’ outputs

The combined prediction is the class obtaining a weighted majority-vote, where votes are weighted with the predictor weights $W_t$.

```{math}
:label: 
\hat{y}(x^\text{new}) = \arg\max_k \; \sum_{t = 1}^{N^\text{pred}} W_t \; \big[ \hat{y}_t(x^\text{new}) = k \big]
```

The `argmax` operator returns the value of its argument that maximizes a given function. The expression in square brackets acts as an indicator, counting 1 if the condition is true and 0 otherwise. For each class $k$, we sum over all predictors $t$. The predicted class is the one with the largest weighted sum. 


## Implementation

In Scikit-Learn, the AdaBoost classifier can be implemented this way (sample loading not shown):
```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier( 
    DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    algorithm="SAMME.R",
    learning_rate=0.5)

ada_clf.fit(X_train, y_train)
```

The decision trees are very 'shallow' learners: only a root note and two final leaf nodes (that's what a max depth of 1 translates to). But there are usually a couple of hundreds of them. The `SAMME` acronym stands for Stagewise Additive Modeling using a Multiclass Exponential Loss Function. It's nothing else than an extension of the algorithm where there are more than two classes. The `.R` stands for Real and it allows for probabilities to be estimated. The predictors need the option `predict_proba` activated, otherwise it will not work. The predictor weights $W_t$ can be printed using `ada_clf.estimator_weights_`. 


## References

```{admonition} Learn more
:class: seealso
* [AdaBoost on Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* AdaBoost, Clearly Explained - StatQuest [video on YouTube](https://www.youtube.com/watch?v=LsK-xG1cLYA)
* [Multi-class AdaBoosted Decision Trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html) from scikit-learn.org 
```
