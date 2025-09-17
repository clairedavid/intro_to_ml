# Loss and Cost Functions

A quick refresher of what we saw in the first lectures... with a few small tweaks in notation to fit the neural network world.


## Terminology - General definitions
One should not confuse cost and loss. 

````{prf:definition}
:label: costnndef
The __loss function__ quantifies the difference between the actual and predicted value for one sample instance.

The __cost function__ aggregates the differences of all instances of the dataset. It can have a regularization term.
````

```{warning}
The loss function is not to be confused with the hypothesis function $h_{W,b}(x^{(i)})$ that serves to build a prediction $\hat{y}^{(i)}$ for sample $i$.


This is not to be confused with the activation function either, which only gets the information from the artificial neuron's inputs (data or neurons' output values), and does not perform any comparison with the observed values $\boldsymbol{y}$.
```


## Loss Functions for Regression

### Mean Squared Error (MSE)
The most commonly used loss function is the Mean Squared Error (MSE) that we are now familiar with. If we have only one output node:
```{math}
:label: lossmseeq
L \left(\;\hat{y}^{(i)}, y^{(i)}\;\right)=  \left(  \hat{y}^{(i)} - y^{(i)}  \right)^2
```
where $i$ here is the sample number.

For several output dimensions (e.g. multiple regression targets or classes), we sum the squared differences across all components:

```{math}
:label: lossmsekeq
L \left(\;\boldsymbol{\hat{y}}^{(i)}, \boldsymbol{y}^{(i)}\;\right)= \;\sum_{k = 1}^K  \left( \hat{y}^{(i)}_k - y^{(i)}_k \right)^2
```
with the $k$ indices being the prediction or observed value for the node $k$.

Then the cost function would be the average of the losses over all the training sample of $m$ instances:
```{math}
:label: costmsekeq
C \left(\;\boldsymbol{\hat{y}}, \boldsymbol{y}\;\right)= \frac{1}{m} \sum_{i=1}^m \sum_{k = 1}^K  \left( \hat{y}^{(i)}_k - y^{(i)}_k \right)^2
```



### Absolute Loss
If there are lots of outliers in the training set, aka samples associated with a large error between the prediction and the observed values, the Mean Squared Errror will make the loss (and cost) very big. A preferable choice would be to take the absolute loss:
```{math}
:label: lossabseq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)= \left| \;\hat{y}_k^{(i)} - y_k^{(i)} \; \right|
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
L \left(\;\hat{y}^{(i)}, y^{(i)}\;\right)=-\left[ \;y^{(i)}  \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right) \; \right]
```
And the cost function will be

```{math}
:label: costbinceeq
C \left(\;\hat{y}, y\;\right) = - \frac{1}{m} \sum_{i=1}^m \left[ \;y^{(i)}  \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right) \; \right] 
```

````{margin}
Recall that $\hat{y}^{(i)} = h_{\boldsymbol{W},\boldsymbol{b}}(\boldsymbol{x}^{(i)})$.
````
There is nothing new here, except that the predictions $\boldsymbol{\hat{y}}^{(i)}$ from the hypothesis function are not a linear function but the output of the entire neural network forward propagation.


### Categorical Cross-Entropy
A neural network with one output node will classify from two classes: $y=1$ and $y=0$. The cross-entropy is the sum of the actual outcome multiplied by the logarithm of the outcome predicted by the model. If we have more than two classes, we can write the outcome of a given training data instance $i$ as a vector:
```{math}
:label: catvecobseq
\boldsymbol{y}^{(i)} = ( 1, 0, 0, \cdots 0)
```
of K elements, where K is the number of output nodes. A sample belonging to the class $k$ corresponds to the row index $k$. For instance a sample of the second class (the order is to be defined by convention) would be $\boldsymbol{y}^{(i)} = (0, 1, 0, \cdots 0 )$. 

A multi-class neural network would produce vectorial predictions for one sample of the form:
```{math}
:label: ypredmulticlasseq
\boldsymbol{\hat{y}}^{(i)} = (0.15, 0.68, \cdots , 0.03)
```
````{margin}
The categorical cross-entropy is appropriate in combination with an activation function such as the softmax that can produce several probabilities for the number of classes that sum up to 1.
````
Mutually exclusive classes would mean that for each $\boldsymbol{\hat{y}}^{(i)} = (\hat{y}^{(i)}_1, \hat{y}^{(i)}_2, \cdots, \hat{y}^{(i)}_K)$, all output values should add up to 1:
```{math}
:label: ymulticlassoneeq
\sum_{k=1}^K \; \hat{y}^{(i)}_k = 1
```
````{margin}
This is the general equation for $K$ classes.
````
The categorial cross-entropy reduces to the binary equation {eq}`lossbinceeq` for $K =2$.
```{math}
:label: losscateq
L \left(\;\hat{y}^{(i)}_k, y^{(i)}_k\;\right)=  - \sum_{k=1}^K \; y^{(i)}_k \log \left( \hat{y}_k^{(i)} \right) 
```

And the cost function becomes:

```{math}
:label: costcateq
C \left(\;\boldsymbol{\hat{y}}, \boldsymbol{y}\;\right) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \; y^{(i)}_k \log \left( \hat{y}^{(i)}_k \right) 
```


We’re done with the anatomy of our little neurons. Now let's see how to train them!