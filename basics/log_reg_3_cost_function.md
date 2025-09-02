# Cost Function for Classification



## Wavy least squares
If we plug our sigmoid hypothesis function $h_\boldsymbol{\theta}(x)$ into the cost function defined for linear regression (Equation {eq}`costFunctionLinReg` from Lecture {ref}`Cost Function for Linear Regression<linReg:Cost>`), we will have a complex non-linear function that could be non-convex. The cost function could take this form: 

```{glue:figure} poly3minima_example
:name: "poly3minima_example"
:figwidth: 80%
```

Imagine running a gradient descent procedue that starts from a randomly initialized $\theta_0$ parameter around zero (or worse, lower than -2). It will fall into a local minima. Our cost function will not be at the global minimum! It is crucial to work with a cost function accepting one unique minimum.


## Building a new cost function
As we saw in the previous section, the sigmoid fits the 1D data distribution very well. Our cost function will use the hypothesis $h_\boldsymbol{\theta}(\boldsymbol{x})$ function as input. Recall that the hypothesis $h_\boldsymbol{\theta}(\boldsymbol{x})$ is bounded between 0 and 1. What we need is a cost function producing high values if we mis-classify events and values close to zero if we correctly label the data. Let's examine what we want for the two cases:

__Case of a signal event:__  
A data point labelled signal verifies by our convention $y=1$. If our hypothesis $y^\text{pred} = h_\boldsymbol{\theta}(\boldsymbol{x})$ is also 1, then we have a good prediction. The cost value should be zero. If however our signal sample has a wrong prediction $y^\text{pred} = h_\boldsymbol{\theta}(\boldsymbol{x}) = 0$, then the cost function should take large values to penalize this bad prediction. We need thus a strictly decreasing function, starting with high values and cancelling at the coordinate (1, 0). 

__Case of a background event:__  
The sigmoid can be interpreted as a probability for a sample being signal or not (but note it is not a probability distribution function). As we have only two outcomes, the probability for a data point to be non-signal will be in the form of $1 - h_\boldsymbol{\theta}(\boldsymbol{x})$. We want to find a function with this time a zero cost if the prediction $y^\text{pred} = h_\boldsymbol{\theta}(\boldsymbol{x}) = 0$ and a high cost for an erroneous prediction $y^\text{pred} = h_\boldsymbol{\theta}(\boldsymbol{x}) = 1$.

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
The __cost function for logistic regression__ is defined as:
```{math}
:label: costFunctionLogReg
J(\boldsymbol{\theta}) = - \frac{1}{m} \sum^m_{i=1} \left[ \;\; {\color{RoyalBlue}y^{(i)} \log\left( h_\boldsymbol{\theta}(\boldsymbol{x}^{(i)} ) \right) }\;\;+\;\; {\color{OliveGreen}(1- y^{(i)}) \log\left( 1 - h_\boldsymbol{\theta}(\boldsymbol{x}^{(i)} ) \right)} \;\;\right]
 ```
 This function is also called __cross-entropy loss function__ and is the standard cost function for binary classifiers.
````

Note the negative sign factorized at the beginning of the equation. The first and second term inside the sum are multiplied by ${\color{RoyalBlue}y^{(i)}}$ and ${\color{OliveGreen}(1 - y^{(i)})}$. This acts as a "switch" between the two possible cases for the targets: ${\color{RoyalBlue}y=1}$ and ${\color{OliveGreen}y=0}$. If ${\color{RoyalBlue}y=1}$, the second term cancels out and the cost takes the value of the first. If ${\color{OliveGreen}y=0}$, the first term vanishes. The two mutually exclusive cases are combined into one mathematical expression. 


