# Bias & Variance: a Tradeoff

The bias and variance are the mathematical underpinnings of two distinct scenarios of mis-fit: underfitting and overfitting. Those situations are symptoms of high bias and high variance, respectively. Let's first intuitively introduce under- and overfitting, then see how to diagnose them with convenient plots before presenting the bias-variance tradeoff. Last but not least, we will explore strategies to mitigate misfit and guide the model toward the optimal “sweet spot.”

## Underfitting, overfitting

Let's have a look at the three cases below:

```{figure} ../images/modEval_underoverfit.png
---
  name: modEval_underoverfit
  width: 1200px
---
 . Example of several regression models attempting to fit data points (blue dots) generated from a true function (orange curve). The fit is depicted in blue for a polynomial of degree 1 (left), degree 4 (middle) and degree 15 (right). 
  
 <sub>Source: [scikit-learn.org (with associated python code)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)</sub>
```

What can we say qualitatively about the quality of those fits? The linear one on the left (polynomial of degree 1) is clearly missing out the main pattern of the data. No matter the slope and the offset, a straight line will never fit a wavy curve decently. This is a case of underfitting. The model is too simple. On the other hand, the fit on the right (polynomial of degree 15) is perfectly passing through all datapoints. Technically, the performance is excellent! But the model is abusing its numerous degrees of freedom and has done more than fitting the data: it captured all the fluctuations and the noise specific to the given dataset. If we regenerate samples from the orange true function, the blue curve with large oscillations will definitely not pass through the newly generated points. There will be substantial errors. The excess of freedom with the high-degree polynomial is a bit of a curse here. This is a case of overfitting: the model is over-specific to the given random variations in the dataset. In the middle model, we seem to find a good compromise with a polynomial function of degree 4.

 Hope this gives you a feel of the under and overfitting (under and overtraining, synonym). Now let's write the definitions!


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

Overfitting is synonym of overtuned, overtweaked. In other words: the model learns the detail and noise in the training dataset to the extent that it negatively impacts the performance of the model on a new dataset. This means that the noise or random fluctuations in the training dataset is picked up and learned as concepts by the model. 
In machine learning, we look for trends and compromises: a good algorithm may not be perfectly classifying a given data set; it needs to accommodate and ignore rare outliers so that future predictions, on average, will be accurate (we will see how to diagnose learners in the next section).

The problem with overfitting is the future consequences once the machine learning algorithm receives additional data: it may lack flexibility.

````{prf:definition}
:label: flexibilitydef
The __flexibility__ of a model determines its ability to generalize to different characteristic of the data.
````
In some definitions (it seems there is no standard definition of flexibility), the literature quotes "to increase the degrees of freedom available to the model to fit to the training data." What are degrees of freedom in this context? Think of data points distributed along a parabola. A linear model will be underfitting the data as it is too simple to catch the parabolic trend with only two degrees of freedom (remember there are two parameters to optimize). A quadratic equation, however, will manage well with three degrees of freedom. A model with more degrees of freedom has margin to adapt well to different situations. This is the idea behind flexibility. 




## Definitions
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

```{figure} ../images/modEval_bias_var_ierrors.png
---
  name: modEval_bias_var_ierrors
  width: 90%
---
 . Decomposition of the generalized error into the bias, variance and irreducible errors.  
 <sub>Image: [towardsdatascience.com](https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9)</sub>
```
Increasing the model complexity will reduce the bias but increase the variance. Reversely, simplifying a model to mitigate the variance comes at a risk of a higher bias. In the end, the lowest total error is a trade-off between bias and variance.

````{prf:definition}
:label: biasdef
The __bias__ is an error coming from wrong assumptions on the model. 

A highly biased model will most likely underfit the data.

````
The bias implies not grasping the full complexity of the situation (think of a biased person making an irrelevant or indecent remark in a conversation).


````{prf:definition}
:label: variancedef
The __variance__ is a measure of the model's sensitivity to statistical fluctuations of the training data set. 

A model with high variance is likely to overfit the data.

````

As its name suggest, a model incorporating fluctuations in its design will change, aka _vary_, as soon as it is presented with new data (fluctuating differently). 

Using a larger training dataset will reduce the variance.

Below is a good visualization of the two tendencies for both regression and classification:

```{figure} ../images/modEval_underoverfit_reg_class_table.jpg
---
  name: modEval_underoverfit_reg_class_table
  width: 100%
---
 . Illustration of the bias and variance trade-off for regression and classification.  
 <sub>Image: LinkedIn Machine Learning India</sub>
```
Before learning on ways to cope with either bias or variance, we need first to assess the situation. How to know if our model has high bias or high variance?


## Identifying the case
By plotting the cost function with respect to the model's complexity. Increasing complexity can be done by adding more features, higher degree polynomial terms, etc. This implies running the training and validation each time with a different model to collect enough points to make such a graph:

```{figure} ../images/modEval_bias-variance-train-val-complexity.png
---
  name: modEval_bias-variance-train-val-complexity
  width: 90%
---
 . Visualization of the error (cost function) with respect to the model's complexity for the training and validation sets. The ideal complexity is in the middle region where both the training and validation errors are low and close to one another.  
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
.  Interpretation of the error plots as a function of the number of samples in the dataset for low and high bias/variance situations.  
<sub>Images: [dataquest.io](https://www.dataquest.io/blog/learning-curves-machine-learning/)</sub>
```

The presence of a small gap between the train and test errors could appear like a good thing. But it important to quantify the training error and relate it to the desired accuracy: if the error is much higher than the irreducible error, chances are the algorithm is suffering from a high bias. 

The variance is usually spotted by the presence of a significant gap pertaining even if the dataset size $m$ increases, yet closing itself for large $m$ (hint for the following section on to cope with variance: getting more data). 

## Regularization to cope with overtraining

... 

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