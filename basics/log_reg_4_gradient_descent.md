# Gradient Descent for Logistic Regression


The gradient descent for classification follows the same procedure as described in {prf:ref}`GD_algo_multi` in Section {ref}`linRegMulti:gradDesc` with the definition of the cost function from Equation {eq}`costFunctionLogReg` above.


## Derivatives in the linear case

Consider the linear assumption $\boldsymbol{x} \cdot \boldsymbol{\theta}  = \theta_0 + \theta_1 x_1 +  \cdots  + \theta_n x_n$ as input to the sigmoid function $\sigma$. 
The cost function derivatives will take the form:
````{margin}
Recall that $h_\boldsymbol{\theta} (\boldsymbol{x}^{(i)}) =  \sigma( \boldsymbol{x}^{(i)} \, \boldsymbol{\theta}) = \frac{1}{1 + e^{-  \boldsymbol{x}^{(i)} \, \boldsymbol{\theta} }}$ 
````
```{math}
:label: costfderivlin
\frac{\partial}{\partial \theta_j} J(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\boldsymbol{\theta}(\boldsymbol{x}^{(i)}) -  y^{(i)}\right) x_j^{(i)}
```
Wait: we've seen this somewhere! This takes the same form as the derivatives for linear regression, see Equation {eq}`partialDevLinReg` in Section {ref}`linRegMulti:gradDesc`.

```{admonition} Exercise
:class: seealso
To convince yourself, you can derive Equation {eq}`costfderivlin` starting from Equation {eq}`costFunctionLogReg`.   

Hints and help available on demand after class.
```

## Gradient descent: pseudo-code for logistic regression

This is left as an exercise 😉 to check your understanding and practice writing pseudo-code.

```{admonition} Exercise
:class: seealso
Based on the algorithm {prf:ref}`GD_algo_multi` in Section {ref}`linRegMulti:gradDesc`, adapt it for logistic regression and write the corresponding pseudo-code, highlighting the differences and including the elements specific to classification.
```

You will put this into practice in the second tutorial, where you’ll code a classifier by hand!

## Alternative techniques
Beside logistic regression, other algorithms are designed for binary classification. 

The [Perceptron](https://en.wikipedia.org/wiki/Perceptron), which is a single layer neural network with, in its original form, a step function instead of a sigmoid function. This will be introduced in the lecture on neural networks.

[Support Vector Machines (SVMs)](https://en.wikipedia.org/wiki/Support-vector_machine) are robust and widely used in classification problems. We will not cover them here but below are some links for further reading.

```{admonition} Learn more
:class: seealso
* R. Berwick, An Idiot’s guide to Support vector machines (SVMs) on [web.mit.edu](https://web.mit.edu/6.034/wwwbob/svm.pdf)
* Support Vector Machines: A Simple Explanation, on [KDNuggets](https://www.kdnuggets.com/2016/07/support-vector-machines-simple-explanation.html)
* Support Vector Machines: Main Ideas" by Josh Starmer, on [StatQuest YouTube channel](https://www.youtube.com/watch?v=efR1C6CvhmE)
 ```

Numerous methods have been developed to find optimized model parameters more efficiently than gradient descent. These optimizers are beyond the scope of this course and usually available as libraries within python (or other languages). Below is a list of the most popular ones:

```{admonition} Learn more
:class: seealso
* The BFGS algorithm: [Wikipedia article](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
* The Limited-memory BFGS, L-BFGS: [Wikipedia article](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
* Conjugate gradient method: [Wikipedia article](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
* [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain, Jonathan Richard Shewchuk (1994)](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)