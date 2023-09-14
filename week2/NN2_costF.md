# Neural Network Loss Function

&nbsp;  
Before getting to the heart of the backpropagation, let's cover an important ingredient: the cost function of a neural network.

This is already a familiar concept since Lecture 2. But there will be some tweaks in the writing to adapt it to the neural network business.

## Terminology - General definitions
One should not confuse cost and loss.

````{prf:definition}
:label: costnndef
The __loss function__ $\boldsymbol{L}$ quantifies the difference between the actual and predicted value for one sample instance.

The __cost function__ $\boldsymbol{J}$ aggregates the differences of all instances of the dataset. It can have a regularization term.
````

```{warning}
The loss function is not to be confused with the hypothesis function $h_{W,b}(x^{(i)})$ that serves to build a prediction $\hat{y}^{(i)}$ for sample $i$.


This is not to be confused with the activation function either, which only gets the neural network inputs (training data, weights and biases) and does not perform any comparison with the observed values $\boldsymbol{y}$).
```

We saw that forward propagation 'fills' in the network with values for all activation units using the weight matrices $W^1, W^2, \cdots, W^{L}$ and biases $b^1, b^2, \cdots b^L$, up to the last layer $L$ being the output layer:
```{math}
:label: aLasteq
\boldsymbol{a^{(L)}} = f\left( \; \boldsymbol{a^{(L -1)}} \;W^{(L)} \;+\; \boldsymbol{b}^{(L)} \;\right)
```

If the last layer contains $K$ output nodes, we can write the elements of vector 
```{math}
\boldsymbol{a^{(L)}} = (a^{L}_1, a^{L}_2, \cdots, a^{L}_k, \cdots a^{L}_K)
```
in the form of a hypothesis function. For each output node $a^{L}_k$, the predicted value is:
```{math}
:label: ypredhWbeq
 a^{L}_k = h_{W,b}(x^{(i)})_k  = \hat{y}^{(i)}_k \;,
```
where the hat is the statistical convention for a predicted value (by opposition to an observed one).
It is important to note that in the equation above, the $W,b$ indices refer to __all weights and biases of the network__ (from forward propagation).

## Loss Functions for Regression

### Mean Squarred Error (MSE)
The most commonly used loss function is the Mean Squarred Error (MSE) that we are now familiar with. If we have only one output node:
```{math}
:label: lossmseeq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)= \left(  \hat{y}^{(i)} - y^{(i)}  \right)^2
```
For several output nodes, we sum all losses:

```{math}
:label: lossmsekeq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)= \sum_{k = 1}^K  \left( \hat{y}^{(i)}_k - y^{(i)}_k \right)^2
```
with the $k$ indices being the prediction or observed value for the node $k$.

Then the cost function would be the average of the losses over all the training sample of $m$ instances:
```{math}
:label: costmsekeq
J \left(\;\hat{y}_k, y_k\;\right)= \frac{1}{2m} \sum_{i=1}^m \sum_{k = 1}^K  \left( \hat{y}^{(i)}_k - y^{(i)}_k \right)^2
```


### Absolute Loss
If there are lots of outliers in the training set, aka samples associated with a large error between the prediction and the observed values, the Mean Squarred Errror will make the loss (and cost) very big. A preferrable choice would be to take the absolute loss:
```{math}
:label: lossabseq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)= \left| \;\hat{y}^{(i)} - y^{(i)} \; \right|
```

### Huber Loss
The Huber Loss is a compromise of the two functions above. It is quadratic when the error is smaller than a threshold $\delta$ but linear when the error is larger. The linear part makes it less sensitive to outliers than with MSE. The quadratic part allows it to converge faster and be more precise than the absolute error.
```{math}
:label: losshubereq
L_\delta \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)= 
\begin{cases}
\;\; \frac{1}{2}\;\left(\;\hat{y}^{(i)}_k-y^{(i)}_k \;\right)^2 & \text { for } \left|\; \hat{y}^{(i)}_k-y^{(i)}_k \right| \leq \delta \\[2ex]
\;\; \delta \cdot\left(\;\left|\;\hat{y}^{(i)}_k-y^{(i)}_k\;\right|-\frac{1}{2} \; \delta \; \right), & \text { otherwise }
\end{cases}
```

## Loss Functions for Classification

### Binary Cross-Entropy
We are familiar with this one as it was introduced in Lecture 3. We will rewrite it as a reminder: 

```{math}
:label: lossbinceeq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)=-\left[ \;y^{(i)}  \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right) \; \right]
```
And the cost function will be

```{math}
:label: costbinceeq
J \left(\;\hat{y}_k, y_k\;\right) = - \frac{1}{m} \sum_{i=1}^m \left[ \;y^{(i)}  \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right) \; \right] 
```
There is nothing new here, except that the hypothesis function $h_{W,b}(x^{(i)})$ is not a linear function but the output of neural network forward propagation.

### Categorical Cross-Entropy
A neural network with one output node will classify from two classes: $y=1$ and $y=0$. The cross-entropy is the sum of the actual outcome multiplied by the logarithm of the outcome predicted by the model. If we have more than two classes, we can write the outcome of a given training data instance $i$ as a vector:
```{math}
:label: catvecobseq
\boldsymbol{y^{(i)}} = ( 1, 0, 0, \cdots 0)
```
of K elements, where K is the number of output nodes. A sample belonging to the class $k$ corresponds to the row index $k$. For instance a sample of the second class (the order is to be defined by convention) would be $\boldsymbol{y^{(i)}} = (0, 1, 0, \cdots 0 )$. 

A multi-class neural network would produce vectorial predictions for one sample of the form:
```{math}
:label: ypredmulticlasseq
\boldsymbol{\hat{y}^{(i)}} = (0.15, 0.68, \cdots , 0.03)
```
````{margin}
The categorical cross-entropy is appropriate in combination with an activation function such as the softmax that can produce several probabilities for the number of classes that sum up to 1.
````
Mutually exclusive classes would mean that for each $\boldsymbol{\hat{y}^{(i)}} = (\hat{y}^{(i)}_1, \hat{y}^{(i)}_2, \cdots, \hat{y}^{(i)}_K)$, all output values should add up to 1:
```{math}
:label: ymulticlassoneeq
\sum_{k=1}^K \; \hat{y}^{(i)}_k = 1
```
````{margin}
This is the general equation for $K$ classes.
````
The categorial cross-entropy reduces to the binary equation {eq}`lossbinceeq` for $n =2$.
```{math}
:label: losscateq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)=  - \sum_{k=1}^K \; y^{(i)}_k \log \left( \hat{y}^{(i)} \right) 
```

And the cost function

```{math}
:label: costcateq
J \left(\;\hat{y}_k, y_k\;\right) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \; y^{(i)}_k \log \left( \hat{y}^{(i)}_k \right) 
```

## Regularization term
In Lecture 3 section {ref}`class:algs:reg` we saw two techniques adding an extra penalty in order to constrain the $\theta$ parameters, helping at mitigating overtraining. Here the regularization term here will be expressed using the neural network $W$ matrices.

```{math}
:label: 
J \left(\;\hat{y}_k, y_k\;\right) = \frac{1}{m} \sum_{i=1}^m  L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right) + \frac{\lambda}{2m} \sum_{\ell=1}^L \sum_{q=1}^{n^{L-1}} \sum_{r=1}^{n^{L}} \left( W^{(\ell)}_{q, r} \; \right)^2
```

The triple sum is daunting but let's decompose it: it's the sum of all matrices $W^{(\ell)}$ of the network. For each of them, the weights are squared and added together (those are the two last sums). The equation above corresponds to the Lasso's regularization (introduced in Lecture 3 subsection {ref}`class:algs:reg:lasso`). The Ridge regularization would sum only the weights without squaring them.

Recall that the regularization does not include the intercept term, which was written as $\theta_0$ in logistic regression. With the matrices $W^{(\ell)}$ we are safe as they do not contain any bias terms. Biases are gathered as separate vectors $\boldsymbol{b^{(\ell)}}$. 

&nbsp;

We've seen forward propagation in the previous lecture and here the loss functions for neural network. One more thing before heading to the backpropagation:
 
__<center>How are the weights initialized?</center>__