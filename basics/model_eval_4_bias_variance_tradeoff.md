# Bias & Variance: a Tradeoff

The bias and variance are the mathematical underpinnings of two distinct scenarios of mis-fit: underfitting and overfitting. Those situations are symptoms of high bias and high variance, respectively. Let's first intuitively introduce under- and overfitting, then see how to diagnose them with convenient plots before presenting the bias-variance tradeoff. Last but not least, we will explore strategies to mitigate misfit and guide the model toward the optimal “sweet spot.”

## Underfitting, overfitting

Let's have a look at the three cases below:

```{figure} ../images/modEval_underoverfit.png
---
  name: modEval_underoverfit
  width: 1200px
---
: Example of several regression models attempting to fit data points (blue dots) generated from a true function (orange curve). A fitting attempt is depicted in blue for a polynomial of degree 1 (left), degree 4 (middle) and degree 15 (right). 
  
 <sub>Source: [scikit-learn.org (with associated python code)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)</sub>
```

What can we say qualitatively about the quality of those fits? The linear one on the left (polynomial of degree 1) is clearly missing out the main pattern of the data. No matter the slope and the offset, a straight line will never fit a wavy curve decently. This is a case of underfitting. The model is too simple. On the other hand, the fit on the right (polynomial of degree 15) is perfectly passing through all datapoints. Technically, the performance is excellent! But the model is abusing its numerous degrees of freedom and has done more than fitting the data: it captured all the fluctuations and the noise specific to the given dataset. If we regenerate samples from the orange true function, the blue curve with large oscillations will definitely not pass through the newly generated points. There will be substantial errors. The excess of freedom with the high-degree polynomial is a bit of a curse here. This is a case of overfitting: the model is over-specific to the given random variations in the dataset. In the middle model, we seem to find a good compromise with a polynomial function of degree 4.

 Hope this gives you a feel of the underfitting and overfitting (or undertraining and overtraining, synonyms). Now let's write the definitions!


````{prf:definition}
:label: underfittingdef
__Underfitting__ is a situation that occurs when a fitting procedure or machine learning algorithm is not capturing the general trend of the dataset.  
````
Another way to put it: an underfit algorithm lacks complexity.

However, if we add too much complexity, we can fall in another trap: overfitting.

````{prf:definition}
:label: overfittingdef
__Overfitting__ is a situation that occurs when a fitting procedure or machine learning algorithm matches too precisely a particular collection of a dataset. 
````

Overfitting is synonym of overtuned, overtweaked. In other words: the model learns the details and noise in the training dataset to the extent that it negatively impacts the performance of the model on a new dataset. This means that the noise or random fluctuations in the training dataset are picked up and learned as actual trends by the model. 
In machine learning, we look for general patterns and compromises: a good algorithm may not be perfectly classifying a given data set; it needs to accommodate and ignore rare outliers so that future predictions, on average, will be accurate.

The problem with overfitting is the future consequences once the machine learning algorithm receives additional data: it will very likely fail to generalize to new data.  


````{prf:definition}
:label: generalizationdef
In machine learning, generalization is the ability of a trained model to perform well on new, unseen data drawn from the same distribution as the training data.
````

````{prf:definition}
:label: capacitydef
The __capacity__ of a model determines its ability to remember information about it's training data.
````

Capacity is not a formal term, but it is linked to the number of degrees of freedom available to the model. A linear fit (two degrees of freedom) is limited to match a parabola. The model lacks capacity. However too much capacity can backfire! As in our example below, a model with many degrees of freedom will be so free that it is likely to fit some fluctuations. Problem. 

But there is a trick! Let's have those degrees of freedom (you get that with low-degree polynomials, we are stuck) but let's add a constraint so as to tame the overfitting. This is called regularization. 


## Regularization to cope with overtraining

In {numref}`modEval_underoverfit`, the model on the right that overfits the data is extremely wiggly. To produce such sharp peaks, some of the polynomial’s coefficients (the model parameters) must take on very large values to “force” the curve through the points. What if we added the values of the model parameters (the $\boldsymbol{\theta}$ components) to the cost function? Since the cost is to be minimized, this could discourage the curve from becoming too wiggly. This is the idea behind the most common regularization techniques.

````{prf:definition}
:label: regularizationdef
__Regularization__ in machine learning is a processus consisting of adding constraints on a model's parameters.  
````

The two main types of regularization techniques are the Ridge Regularization (also known as [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization), albeit the latter is more general) and the Lasso Regularization. 

### Ridge Regression
The Ridge regression is a linear regression with an additional regularization term added to the cost function:
```{math}
:label: ridgeeq
 J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) -  y^{(i)}\right)^2 + {\color{Maroon}\lambda \sum_{j=1}^n \theta_j^2} \right]
```
The hyperparameter $\lambda$ controls the degree of regularization. If $\lambda = 0$, the regularization term vanishes and we have a non-regularized linear regression. You can see the penalty imposed by the term $\lambda$ will force the parameters $\theta$ to be as small as possible; this helps avoiding overfitting. If $\lambda$ gets very large, the parameters can be so shrinked that the model becomes over-simplified to a straight line and thus underfit the data.

```{note}
The factor $\frac{1}{2}$ is used in some derivations of the regularization. This makes it easier to calculate the gradient, however it is only a constant value that can be compensated by the choice of the parameter $\lambda$.
```

```{warning}
The offset parameter $\theta_0$ is not entering in the regularization sum. 
```
In the litterature, the parameters are denoted with $b$ for the offset (bias) and a vector of weight $\vec{w}$ for the other parameters $\theta_1, \theta_2, \cdots \theta_n$. Thus the regularization term is written:

```{math}
:label: regl2weq
\lambda \left(\left\| \vec{w} \right\|_2\right)^2
```
````{margin}
The $\ell_2$ norm is the Euclidian norm $\left\| x \right\|_2 = \sqrt{x_0^2 + x_1^2 + \cdots + x_n^2}$.
````
with $\left\| \vec{w} \right\|_2$ the $\ell_2$ norm of the weight vector.

For logistic regression, the regularized cost function becomes:
```{math}
:label: ridgelogeq
 J(\theta) = - \frac{1}{m} \sum^m_{i=1} \left[ \;\; y^{(i)} \log( h_\theta(x^{(i)} )) \;+\; (1- y^{(i)}) \log( 1 - h_\theta(x^{(i)} )) \;\;\right] + {\color{Maroon}\frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2}
```
(class:algs:reg:lasso)=
### Lasso Regularization
````{margin}
The $\ell_1$ norm is the sum of the magnitudes of the vectors. It is also called Manhattan or [Taxicab norm](https://en.wikipedia.org/wiki/Taxicab_geometry).
````
Lasso stands for least absolute shrinkage and selection operator. Behind the long acronym is a regularization of the linear regression using the $\ell_1$ norm. We denote Cost($\theta$) the cost function, i.e. either the Mean Squared Error for linear regression or the cross-entropy loss function {eq}`costFunctionLogReg` for logistic regression. The lasso regression cost function is
```{math}
:label: lassoCostF
J(\theta) = \text{Cost(}{\theta}\text{)}  + {\color{Maroon}\frac{\lambda}{2m} \sum_{j=1}^n | \theta_j | }
```
The regularizing term uses the $\ell_1$ norm of the weight vector: $\left\| \vec{w} \right\|_1$.


````{warning}
As regularization influences the parameters, it is important to first perform feature scaling before applying the regularization.  
````

__Which regularization method to use?__  

Each one has its pros and cons. As $\ell_1$ (lasso) is a sum of absolute values, it is not differentiable and thus more computationally expensive. Yet $\ell_1$ better deals with outliers (extreme values in the data) by not squaring their values. It is said to be more robust, i.e. more resilient to outliers in a dataset.

The $\ell_1$ regularization, by shriking some parameters to zero (making them vanish and no more influencial), has _feature selection_ built in by design. If this can be advantageous in some cases, it can mishandle highly correlated features by arbitrarily selecting one over the others.

Additional reading are provided below to deepen your understanding in the different regularization methods. Take home message: both methods combat overfitting.

```{admonition} Learn more
:class: seealso
* A comparison of the pros and cons of Ridge ($\ell_2$ norm) and lasso ($\ell_1$ norm) regularization: ["L1 Norms versus L2 Norms", Kaggle](https://www.kaggle.com/code/residentmario/l1-norms-versus-l2-norms/notebook)
* [Fighting overfitting with $\ell_1$ or $\ell_2$ regularization, neptune.ai](https://neptune.ai/blog/fighting-overfitting-with-l1-or-l2-regularization)
```


## Bias & Variance: definitions

Underfitting and overfitting are symptoms of a fundamental modeling issue. This tension is captured by the notions of bias and variance. We will define them, see how they combine in the total error, and why it is not possible to reduce both simultaneously. This is the infamous dilemma of the bias–variance tradeoff.

### Bias
````{prf:definition}
:label: biasdef
The __bias__ is an error coming from wrong assumptions on the model. 

A highly biased model will most likely underfit the data.

````
The bias implies not grasping the full complexity of the situation (think of a biased person making an irrelevant or indecent remark in a conversation).

### Variance

````{prf:definition}
:label: variancedef
The __variance__ is a measure of the model's sensitivity to statistical fluctuations of the training data set. 

A model with high variance is likely to overfit the data.

````

As its name suggest, a model incorporating fluctuations in its design will change, aka _vary_, as soon as it is presented with new data (fluctuating differently). 

Using a larger training dataset will reduce the variance.

### Illustratively

Below is a good visualization of the two tendencies for both regression and classification:

```{figure} ../images/modEval_underoverfit_reg_class_table.jpg
---
  name: modEval_underoverfit_reg_class_table
  width: 100%
---
: Illustration of situations of high bias (left) and high variance (right) for regression and classification.  
 <sub>Image: LinkedIn Machine Learning India</sub>
```



## Generalization error

### Graphically

If we plot the test error (reminder: on a dataset not used for training) with respect to the model's complexity, we can easy see how the bias is the main driver of the error at low model's compl



```{figure} ../images/modEval_bias_var_ierrors.png
---
  name: modEval_bias_var_ierrors
  width: 90%
---
: Decomposition of the generalized error into the bias, variance and irreducible errors.  
 <sub>Image: [towardsdatascience.com](https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9)</sub>
```
Increasing the model complexity will reduce the bias but increase the variance. Reversely, simplifying a model to mitigate the variance comes at a risk of a higher bias. In the end, the lowest total error is a trade-off between bias and variance.







The generalization error can be expressed as a sum of three errors:
```{math}
:label:
\text{Expected Test Error} =
\underbrace{\text{Bias}^2}_{\text{systematic error}}
+
\underbrace{\text{Variance}}_{\text{sensitivity to data}}
+
\underbrace{\sigma^2}_{\text{irreducible noise}}

```
The two first are _reducible_. In fact, we will see in the following how to reduce them as much as possible! The last one is due to the fact data is noisy itself. It can be minimized during data cleaning by removing outliers (or more upfront by improving the detector or device that collected the data). 



Before learning on ways to cope with either bias or variance, we need first to assess the situation. How to know if our model has high bias or high variance?  
## Identifying the case
By plotting the cost function with respect to the model's complexity. Increasing complexity can be done by adding more features, higher degree polynomial terms, etc. This implies running the training and validation each time with a different model to collect enough points to make such a graph:

```{figure} ../images/modEval_bias-variance-train-val-complexity.png
---
  name: modEval_bias-variance-train-val-complexity
  width: 90%
---
: Visualization of the error (cost function) with respect to the model's complexity for the training and validation sets. The ideal complexity is in the middle region where both the training and validation errors are low and close to one another.  
 <sub>Image: [David Ziganto](https://dziganto.github.io/cross-validation/data%20science/machine%20learning/model%20tuning/python/Model-Tuning-with-Validation-and-Cross-Validation/)</sub>   
 ```

It can impractical to test several models with higher complexity. More achievable graphs would be to plot the error (cost function) with respect to the sample size $m$ or the number of epochs $N$:

```{figure} ../images/modEval_low_high_bias.webp
---
  name: modEval_low_high_bias
  width: 100%
---
```
```{figure} ../images/modEval_low_high_var.webp
---
  name: modEval_low_high_var
  width: 100%
---
: Interpretation of the error plots as a function of the number of samples in the dataset for low and high bias/variance situations.  
<sub>Images: [dataquest.io](https://www.dataquest.io/blog/learning-curves-machine-learning/)</sub>
```

The presence of a small gap between the train and test errors could appear like a good thing. But it important to quantify the training error and relate it to the desired accuracy: if the error is much higher than the irreducible error, chances are the algorithm is suffering from a high bias. 

The variance is usually spotted by the presence of a significant gap pertaining even if the dataset size $m$ increases, yet closing itself for large $m$ (hint for the following section on to cope with variance: getting more data). 


## How to deal with bias or variance

The actions to perform to mitigate either bias or variance once we have diagnosed the situation can be done on the dataset, on the model itself and on the regularization. The table below summarizes the relevant treatments to further optimize your machine learning algorithm in the good direction.

```{list-table}
:header-rows: 1

* - Action categories
  - Reducing Bias
  - Reducing Variance
* - On the dataset
  - 
  - Adding more (cleaned) data
* - On the model
  - Adding new features and/or polynomial features
  - Reducing the number of features
* - Regularization
  - Decreasing parameter $\lambda$
  - Increasing parameter $\lambda$
```
  

The tutorials will offer a good training for you (and validation 😉) to diagnose and correctly optimize your machine learning algorithm. 

```{admonition} Learn more
:class: seealso
* [Confusion Matrix, Scikit-Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) 
* [Machine Learning Model Performance and Error Analysis, LinkedIn](https://www.linkedin.com/pulse/machine-learning-model-performance-error-analysis-payam-mokhtarian)

```