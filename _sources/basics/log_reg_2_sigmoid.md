(class:sigmoid)=
# What is the Sigmoid Function?

## General definition

A sigmoid function refers in mathematics to a category of functions having a characteristic "S-shape" curve. Among numerous examples, the one commonly used in machine learning is the logistic function.

````{margin}
To avoid confusion with the fact we already use $x$ for the input features, the input variable in the definition opposite is written with another letter, $z$.
````

````{prf:definition}
:label: logFuncDef
The __logistic function__ is defined for $z \in \mathbb{R}$ as 

\begin{equation*}
\sigma(z) = \frac{1}{1 + e^{-z}}
\end{equation*}
````

It looks like this:

```{glue:figure} sigmoid_example
:name: "sigmoid_example"
```

(class:sigmoid:better)=
## A better mapping

In our example with the energy of the electron, we could see from the data (easily because it is only in one dimension) that the bigger the energy of the electron, the more likely it is for the event to be classified as signal. If we overlay the S-curve on the data points, we start seeing interesting things.

```{figure} ../images/logReg_scatter1D_sigmoid.png
---
name: scattersigmoid
scale: 70%
align: center
---
: Data 1D distribution and sigmoid overlaid
```  
First of all, the curve is not overshooting below or above our discrete outcomes' range. Second: for data points either far left or far right, instead of creating a large error with a straight line as previously, the S-curve actually takes the values of our target-variables (asymptotically). Consequence: the error between the prediction and observed values will be very small, even negligible. We will not have an unwanted shift and mis-classification like before. 


## Sigmoid function for logistic regression

````{prf:definition}
:label: mappingSigDef

A __mapping function__ $h_\theta(x)$ used for logistic regression is of the form:

\begin{equation*}
h_\theta (\boldsymbol{x}^{(i)}) =  \sigma(\boldsymbol{x}^{(i)}\, \boldsymbol{\theta} ) = \frac{1}{1 + e^{- \boldsymbol{x}^{(i)}\, \boldsymbol{\theta} }}
\end{equation*}

where vector $\boldsymbol{x}^{(i)}$ are the input features and $\boldsymbol{\theta}$ the parameters to optimize.  
````

The mapping function satisfies
```{math}
:label: hthetabounded
0 < h_\boldsymbol{\theta} (\boldsymbol{x}^{(i)}) < 1
```
Those limits are reached asymptotically reaching 0 and 1 when $\boldsymbol{x}^{(i)}\, \boldsymbol{\theta}  \rightarrow -\infty$ and $\boldsymbol{x}^{(i)}\, \boldsymbol{\theta}  \rightarrow +\infty$ respectively.

Intuitively, we see in our example that events with very low electron energy are most likely to be background whereas events with high electron energy are more likely to be signal. In the middle, there is a 50/50 chance to mis-classify an event.

__The output of the sigmoid can be interpreted as a probability__. 


## Decision making

Imagine we have optimized our vector $\boldsymbol{\theta}$ parameters. How do we make a prediction?
Previously with linear regression, we could predict the target $y^\text{pred}$ from a new data variable $\boldsymbol{x}^\text{new}$ (a row of all features) by directly using the hypothesis function:
```{math}
:label: ypredthetaxnew
y^\text{pred} = h_\boldsymbol{\theta}(\boldsymbol{x}^\text{new}) =  \boldsymbol{x}^\text{new} \boldsymbol{\theta}
```

With discrete outcomes, we need a new concept to map the input to a discrete class: the decision boundary. 
````{prf:definition}
:label: decBoundDef
A __decision boundary__ is a defined threshold (line, or plane, or more dimensional set of values) created by classifiers to discriminate between the different classes.
````

In the section {ref}`class:sigmoid:better` above, we can split the data sample using $\sigma (z) = 1/2$ horizontal mark:  
* if the logistic function outputs a value $y^\text{pred} < 0.5$, the event is classified as background 
* if the logistic function outputs a value $y^\text{pred} \geq 0.5$, the event is classified as signal

This way and looking at the distribution on {numref}`scattersigmoid`, the majority of data points will be correctly classified with the $1/2$ horizontal threshold.

```{warning}
Careful with how the logistic function is used. We are not computing it directly with $\boldsymbol{x}^{(i)}$ as argument but we calculate $\sigma(\boldsymbol{x}^{(i)}\, \boldsymbol{\theta})$.
```

```{admonition} Question
:class: seealso
How does this boundary translates for sigmoid's input $z = \boldsymbol{x}^\text{new} \, \boldsymbol{\theta} $?  

Look at the first figure on top of this section. For which input values $z$ is the sigmoid lower than 0.5? For which values is the sigmoid above?  
```

````{admonition} Answer
:class: tip, dropdown 

The logistic function $\sigma(z)$ verifies:
* $\sigma(z) < 0.5$ for $z < 0$
* $\sigma(z) \geq 0.5$ for $z \geq 0$

Thus, to predict if a new data point $\boldsymbol{x}^\text{new}$ (row vector of input features) is background of signal, the decision boundary is simply the sign of $\boldsymbol{x}^\text{new} \, \boldsymbol{\theta}$
* if $\boldsymbol{x}^\text{new} \, \boldsymbol{\theta} < 0 \Rightarrow$ the event is classified as background
* if $\boldsymbol{x}^\text{new} \, \boldsymbol{\theta} \geq 0 \Rightarrow$ the event is classified as signal

````

```{note}
Here the input does not have to be linear. We can have any function of the $\boldsymbol{\theta}$ parameters and $\boldsymbol{x}^{(i)}$. In this lecture, we keep the assumption of linearity.
```



With this new tool at hand, let's now see how it is incorporated in a custom-made cost function for logistic regression.









