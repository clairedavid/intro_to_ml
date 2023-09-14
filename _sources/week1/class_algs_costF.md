# Cost Function for Classification


## Wavy least squares
If we plug our sigmoid hypothesis function $h_\theta(x)$ into the cost function defined for linear regression (Equation {eq}`costFunctionLinReg` from Lecture 2), we will have a complex non-linear function that could be non-convex. The cost function could take this form: 

```{glue:figure} poly3minima_example
:name: "poly3minima_example"
:figwidth: 80%
```

Imagine running gradient descent starting from a randomly initialized $\theta_0$ parameter around zero (or worse, lower than -2). It will fall into a local minima. Our cost function will not be at the global minimum! It is crucial to work with a cost function accepting one unique minimum.


## Building a new cost function
As we saw in the previous section, the sigmoid fits the 1D data distribution very well. Our cost function will use the hypothesis $h_\theta(x)$ function as input. Recall that the hypothesis $h_\theta(x)$ is bounded between 0 and 1. What we need is a cost function producing high values if we mis-classify events and values close to zero if we correctly label the data. Let's examine what we want for the two cases:

__Case of a signal event:__  
A data point labelled signal verifies by our convention $y=1$. If our hypothesis $h_\theta(x)$ is also 1, then we have a good prediction. The cost value should be zero. If however our signal sample has a wrong prediction $h_\theta(x) = 0$, then the cost function should take large values to penalize this bad prediction. We need thus a strictly decreasing function, starting with high values and cancelling at the coordinate (1, 0). 

__Case of a background event:__  
The sigmoid can be interpreted as a probability for a sample being signal or not (but note it is not a probability distribution function). As we have only two outcomes, the probability for a data point to be non signal will be in the form of $1 - h_\theta(x)$. We want to find a function with this time a zero cost if the prediction $h_\theta(x) = 0$ and a high cost for an erroneous prediction $h_\theta(x) = 1$.

Now let's have a look at these two functions:

```{glue:figure} log_h_x
:name: "log_h_x"
:figwidth: 100%
```

For each case, the cost function has only one minimum and harshly penalizes wrong prediction by blowing up at infinity.  
How to combine these two into one cost function for logistic regression?  
Like this:

````{prf:definition}
:label: costFLogRegDef
The __cost function for logistic regression__ is a defined as:
```{math}
:label: costFunctionLogReg
J(\theta) = - \frac{1}{m} \sum^m_{i=1} \left[ \;\; {\color{RoyalBlue}y^{(i)} \log( h_\theta(x^{(i)} )) }\;\;+\;\; {\color{OliveGreen}(1- y^{(i)}) \log( 1 - h_\theta(x^{(i)} ))} \;\;\right]
 ```
 This function is also called __cross-entropy loss function__ and is the standard cost function for binary classifiers.
````

Note the negative sign factorized at the beginning of the equation. Multiplying by ${\color{RoyalBlue}y^{(i)}}$ and ${\color{OliveGreen}(1 - y^{(i)})}$ the first and second term of the sum respectively acts as a "switch" between the cases ${\color{RoyalBlue}y=1}$ and ${\color{OliveGreen}y=0}$. If ${\color{RoyalBlue}y=1}$, the first term cancels out and the cost takes the value of the second. If ${\color{OliveGreen}y=0}$, the second term vanishes. The two cases are combined into one mathematical expression.

## Gradient descent
The gradient descent for classification follows the same procedure as described in Algorithm {prf:ref}`GD_algo_multi` in Section {ref}`warmup:linRegMulti:graddesc` with the definition of the cost function from Equation {eq}`costFunctionLogReg` above.

### Derivatives in the linear case
````{margin}
Recall that $h_\theta (x^{(i)}) =  f( x^{(i)}\theta^{T}) = \frac{1}{1 + e^{-  x^{(i)} \theta^{T}}}$ 
````
Consider the linear assumption $x^{(i)}\theta^{\; T}  = \theta_0 + \theta_1 x_1 +  \cdots  + \theta_n x_n$ as input to the sigmoid function $f$. 
The cost function derivatives will take the form:

```{math}
:label: costfderivlin
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) -  y^{(i)}\right) x_j^{(i)}
```
This takes the same form as the derivatives for linear regression, see Equation {eq}`partialDevLinReg` in Section {ref}`warmup:linregmulti:graddesc`.

```{admonition} Exercise
:class: seealso
To convince yourself, you can derive Equation {eq}`costfderivlin` starting from Equation {eq}`costFunctionLogReg`.   

Hints and help available on demand after class.
```

### Alternative techniques
Beside logistic regression, other algorithms are designed for binary classification. 

The [Perceptron](https://en.wikipedia.org/wiki/Perceptron), which is a single layer neural network with, in its original form, a step function instead of a sigmoid function. We will cover neural networks in Lectures 6 and 7.

[Support Vector Machines (SVMs)](https://en.wikipedia.org/wiki/Support-vector_machine) are robust and widely used in classification problems. We will not cover them here but below are some links for further reading.

```{admonition} Learn more
:class: seealso
* R. Berwick, An Idiot’s guide to Support vector machines (SVMs) on [web.mit.edu](https://web.mit.edu/6.034/wwwbob/svm.pdf)
* Support Vector Machines: A Simple Explanation, on [KDNuggets](https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html)
* Support Vector Machines: Main Ideas" by Josh Starmer, on [StatQuest YouTube channel](https://www.youtube.com/watch?v=efR1C6CvhmE)
 ```

Numerous methods have been developed to find optimized $\theta$ parameters in faster ways than the gradient descent. These optimizers are beyond the scope of this course and usually available as libraries within python (or other languages). Below is a list of the most popular ones:

```{admonition} Learn more
:class: seealso
* The BFGS algorithm: [Wikipedia article](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
* The Limited-memory BFGS, L-BFGS: [Wikipedia article](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
* Conjugate gradient method: [Wikipedia article](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
* [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain, Jonathan Richard Shewchuk (1994)](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
```

## More than two categories: multiclassification 
We treated the binary classification problem. How to adapt to a situation with more than two classes?  

Let's for instance consider three classes, labelled with their colours and distributed in two dimensions (two input features) like this:

```{figure} ../images/lec03_3_multiclass-1.webp
---
  name: lec03_3_multiclass-1
  width: 60%
---
 . 2D distribution of three different classes.  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```

A multi-class classification problem can be split into multiple binary classification datasets and be trained as a binary classification model each.
Such approach is a heuristic method, that is to say not optimal nor direct. But it eventually does the job.
There are two main approaches of such methods for multiclassification.

````{prf:definition}
:label: multiclass1to1def

The __One-to-One approach__ consists of applying a binary classification for each pair of classes, ignoring the other classes.

With a dataset made of $N^\text{class}$ classes, the number of models to train, $N^\text{model}$ is given by 
\begin{equation}
N^\text{model} = \frac{N^\text{class}(N^\text{class}-1)}{2}
\end{equation}

Each model predicts one class label. The final decision is the class label receiving the most votes, i.e. being predicted most of the time.
````

__Pro__  
The sample size is more balanced between the two chosen classes than if datasets were split with one class against all others.

__Con__  
The pairing makes the number of models to train large and thus computer intensive.


````{prf:definition}
:label: multiclass1toalldef

The __One-to-All or One-to-Rest approach__ consists of training each class against the collection of all other classes.

With a dataset made of $N^\text{class}$ classes, the number of pairs to train is
\begin{equation}
 N^\text{model} = N^\text{class}
\end{equation}

The final prediction is given by the highest value of the hypothesis function $h^{k}_\theta(x)$, $k \in [1, N^\text{model}]$ among the $N^\text{model}$ binary classifiers.

````
__Pro__  
Less binary classifiers to train.

__Con__  
The number of data points from the class to look for will be very small if the 'background' class is the merging of all other data points from the other classes.


Illustrations. 

The One-to-One method would create those hyperplanes (with two input features, D = 2 we will have a 1D line as separation):

```{figure} ../images/lec03_3_multiclass-2.webp
---
  name: lec03_3_multiclass-2
  width: 60%
---
 . One-to-One approach splits paired datasets, ignoring the points of the other classes.  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```

```{figure} ../images/lec03_3_multiclass-3.webp
---
  name: lec03_3_multiclass-3
  width: 60%
---
 . One-to-All approach focuses on one class to discriminate from all other points  
 (i.e. all other classes are merged into a single 'background' class).  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```

Some further reading if you are curious:

```{admonition} Learn more
:class: seealso
* [One-vs-Rest and One-vs-One for Multi-Class Classification, machinelearningmastery.com](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)
* [Multiclass Classification Using SVM, analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)
```

