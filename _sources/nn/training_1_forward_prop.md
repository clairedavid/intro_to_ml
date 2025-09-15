# Feedforward Propagation

## What is Feedforward Propagation?

This is one of the key steps in the training of a neural network. It comes after initialization of the network (all weights and biases), which we will covered later. The forward direction means going from the input to the output nodes. 

````{prf:definition}
:label: feedforwardpropdef
The __Feedforward Propagation__, also called __Forward Pass__, is the process consisting of computing and storing all network nodes' output values, starting with the first hidden layer until the last output layer, using at start either a subset or the entire dataset samples.
````

Forward propagation thus leads to a list of the neural network predictions for each data instance row used as input. At each node, the computation is the key equation {eq}`aneq` we saw in the previous sub-section {ref}`artif:neuron`, written again for convenience:
```{math}
:label: sumwixieq
\hat{y} = f\left(\sum_{j=1}^n w_j x_j + b \right)
```

Let's define everything in the next subsection.


## Notations
Let's say we have the following network with $x_n$ input features, one first hidden layer with $q$ activation units and a second one with $r$ activation units. For simplicity, we will choose an output layer with only one node:


```{figure} ../images/training_1_nn_notations.png
---
  name: training_1_nn_notations
  width: 100%
---
 . A feedforward neural network with the notation we will use for the forward propagation equations (more in text).    
<sub>Image from the author</sub>
```

There are lots of subscripts and upperscripts here. Let's explain the conventions we will use.  

__Input data__  
We saw in Lecture 2 that the dataset in supervised learning can be represented as a matrix $X$ of $m$ data instances (rows) of $n$ input features (columns). For clarity in the notations, we will focus for now on only one data instance, the $i^{\text{th}}$ sample. We will note it as a column vector $\boldsymbol{x}^{(i)}$: 
```{math}
\boldsymbol{x}^{(i)} = \begin{pmatrix} 
x^{(i)}_1 \\
x^{(i)}_2 \\
\vdots \\
x^{(i)}_n \\
\end{pmatrix}
```
The vector elements are all the features in the data. The upperscript indicates the sample index $i$, going from 1 to $m$, the total number of samples in the dataset.
````{margin}
The layer numbering starts at the first hidden layer where $\ell=1$.
````

__Activation units__  
In a given layer $\ell = 1, 2, \cdots, N^\text{layer}$, the activation units will give outputs that we will note as a column vector as well:
```{math}
\boldsymbol{a}^{(i, \: \ell)} = \begin{pmatrix} 
 a_1^{(i, \: \ell)}\\
a_2^{(i, \: \ell)},\\
\vdots\\
a_q^{(i, \: \ell)} 
\end{pmatrix}  \;, 
```
where subscript is the row of the activation unit in the layer, starting from the top. The upperscript indicates the sample index $i$ and the layer number $\ell$. Why the presence of the sample here? We will see soon that these activation units will get a different value for each data sample. 

__Biases__  
The biases are also column vectors, one for each layer it connects to and of dimension the number of nodes in that layer:
```{math}
\boldsymbol{b}^{(\ell)} = \begin{pmatrix}
b_1^{(\ell)} \\
b_2^{(\ell)} \\
\vdots \\
b_q^{(\ell)}
\end{pmatrix}
```
If the last layer is only made of one node like in our example above, then $b^{(L)}$ is a scalar. Note that the biases do not depend on the sample index $i$. 

__Weights__  
Now the weights. You may see in the literature different ways to represent them. In here we use a convention we could write as:
```{math}
w^\ell_{j \; \to \; k}
```
In other words, the weights from the layer $\ell - 1$ to the layer $\ell$ have as their first index the row of the node from the previous layer (departing node of the weight's arrow). The second index is the row of the node on layer $\ell$ to which the weight arrrow points to. The weight $w^\ell_{j \; \to \; k}$ is the weight departing from node $j$ on layer $\ell - 1$ and connecting node $k$ on layer $\ell$. For instance $w^{(2)}_{3,1}$ is the weight from the third node of layer (1) going to the first node of layer (2). 

We can actually represent each weight from layer $\ell -1$ to layer $\ell$ as a matrix $W^{(\ell)}$. If the previous layer $\ell -1$ has $n$ activation units and the layer $\ell$ has $q$ activation units, we will have:

```{math}
:label: Wmatrixeq
W^{(\ell)} = \begin{pmatrix}
w_{1,1}^{(\ell)} & w_{1,2}^{(\ell)} & \cdots & w_{1,q}^{(\ell)} \\[2ex]
w_{2,1}^{(\ell)} & w_{2,2}^{(\ell)} & \cdots & w_{2,q}^{(\ell)} \\[1ex]
\vdots  & \vdots & \ddots   & \vdots \\[1ex]
w_{n,1}^{(\ell)} & w_{n,2}^{(\ell)} &  \cdots & w_{n,q}^{(\ell)} \\
\end{pmatrix}
```


Note that we do not have an index $i$ for the weight matrix $W^{(\ell)}$. Why? Because the weights are unique for a given network. In fact the weights -- and the biases -- are optimized after the network has incorporated all the data samples. We will actually determine the optimal weights and biases in the next chapter after.

Let's now see how we calculate all the values of the activation units!


## Step by step calculations

### Computation of the first hidden layer 
Let's use Equation {eq}`sumwixieq` to compute the activation unit outputs of the first layer. The activation function is represented as $f$ here. So for a given data sample $i$, we have:
```{math}
:label: firstlayereq
\begin{align*}
a^{(i, \: 1)}_1 &= f\left(\; w_{1,1}^{(1)} \; x^{(i)}_1 \;+\; w_{2,1}^{(1)} \; x^{(i)}_2 \;+\; \cdots + \; w_{n,1}^{(1)} \; x^{(i)}_n \;+\; b^{(1)}_1\right)\\[2ex]
a^{(i, \: 1)}_2 &= f\left(\; w_{1,2}^{(1)} \; x^{(i)}_1 \;+\; w_{2,2}^{(1)} \; x^{(i)}_2 \;+\; \cdots + \; w_{n,2}^{(1)} \; x^{(i)}_n \;+\; b^{(1)}_2\right)\\
&\vdots \\[2ex]
a^{(i, \: 1)}_q &= f\left(\; w_{1,q}^{(1)} \; x^{(i)}_1 \;+\; w_{2,q}^{(1)} \; x^{(i)}_2 \;+\; \cdots + \; w_{n,q}^{(1)} \; x^{(i)}_n \;+\; b^{(1)}_q\right)\\
\end{align*}
```




We can actually write it in the matrix form. Let's first write it in an expanded version with the matrix elements:
```{math}
:label: firstlayermatrixexpandedeq
\boldsymbol{a}^{(i, \: 1)} = f\left[ \; 
\begin{pmatrix}w_{1,1}^{(1)} & w_{1,2}^{(1)} & \cdots & w_{1,q}^{(1)} \\[2ex]w_{2,1}^{(1)} & w_{2,2}^{(1)} & \cdots & w_{2,q}^{(1)} \\[1ex]\vdots  & \vdots & \ddots   & \vdots \\[1ex]w_{n,1}^{(1)} & w_{n,2}^{(1)} &  \cdots & w_{n,q}^{(1)} \\\end{pmatrix}^\top
\begin{pmatrix} 
x^{(i)}_1 \\
x^{(i)}_2 \\
\cdots \\
x^{(i)}_n \\
\end{pmatrix} 
 \;+\; \begin{pmatrix}
b_1^{(1)} \\
b_2^{(1)} \\
\cdots \\
b_q^{(1)}
\end{pmatrix} \; \right]
```
You can verify that $\boldsymbol{a}^{(i, \: 1)}$ will be a column vector with $q$ elements.
This can be written in a compact way:
```{math}
:label: firstlayermatrixeq
\boldsymbol{a}^{(i, \: 1)} = f\left[ \; \left(W^{(1)}\right)^\top \; \boldsymbol{x}^{(i)} \;+\; \boldsymbol{b}^{(1)} \;\right]
```
Much lighter. 

### Computation of the second hidden layer 
Let's do the same calculation for the second layer of activation units. Instead of the dataset vector $\boldsymbol{x}^{(i)}$, we will have $\boldsymbol{a}^{(i, \: 1)}$ as input:
```{math}
:label: secondlayermatrixexpandedeq
\boldsymbol{a}^{(i, \: 2)} = f\left[ \; 
\begin{pmatrix}w_{1,1}^{(2)} & w_{1,2}^{(2)} & \cdots & w_{1,r}^{(2)} \\[2ex]w_{2,1}^{(2)} & w_{2,2}^{(2)} & \cdots & w_{2,r}^{(2)} \\[1ex]\vdots  & \vdots & \ddots   & \vdots \\[1ex]w_{q,1}^{(2)} & w_{q,2}^{(2)} &  \cdots & w_{q,r}^{(2)} \\\end{pmatrix}^\top
\begin{pmatrix} 
a^{(i, \: 1)}_1 \\
a^{(i, \: 1)}_2 \\
\cdots \\
a^{(i, \: 1)}_q \\
\end{pmatrix} 
 \;+\; \begin{pmatrix}
b_1^{(2)} \\
b_2^{(2)} \\
\cdots \\
b_r^{(2)}
\end{pmatrix} \; \right]
```

And the elegant, light version:
```{math}
:label: secondlayermatrixeq
\boldsymbol{a}^{(i, \: 2)} = f\left[ \; \left(W^{(2)}\right)^\top \; \boldsymbol{a}^{(i, \: 1)} \;+\; \boldsymbol{b}^{(2)} \;\right]
```

We start seeing a pattern here looking at the matricial equations {eq}`firstlayermatrixeq` and {eq}`secondlayermatrixeq`. More on this soon in Section {ref}`NN1:forwardprop:rule`. Let's finish the process with the last layer.


### Computation of the third hidden layer 
With one output node, it is actually simpler than for the hidden layers above. We can still write it in the same form as Equation {eq}`secondlayermatrixeq`:
```{math}
:label: thirdlayermatrixeq
\boldsymbol{a}^{(i, \: 3)} = f\left[ \; \left(W^{(3)}\right)^\top \; \boldsymbol{a}^{(i, \: 2)} \;+\; \boldsymbol{b}^{(3)} \;\right]
```
using $\boldsymbol{a}^{(i, \: 2)}$ that we calculated above. In our case $\boldsymbol{a}^{(i, \: 3)}$ has only one element: $a^{(i, \: 3)}_1 = \hat{y}^{(i)}$. Thus the matrix $W^{(3)}$ has only one column. The bias 'vector' is actually a scalar: $b^{(3)}$. 

We have computed a value for each activation unit for a given data sample $\boldsymbol{x}^{(i)}$. That is the end of the forward propagation process! As you can see, it contains lots of calculations. And now you may understand why activation functions that are simple and fast to compute are preferable, as they intervene each time we compute the output of an activation unit.

Let's now get a general formula.

(NN1:forwardprop:rule)=
## General rule for Forward Propagation

If we rewrite the first layer of inputs for a given sample $\boldsymbol{x}^{(i)}$ as a "layer zero" $\boldsymbol{a}^{(i, \: 0)}$:
```{math}
:label: xisazeroeq
\boldsymbol{x}^{(i)} = \begin{pmatrix} 
x^{(i, \: 0)}_1 \\
x^{(i, \: 0)}_2 \\
\cdots \\
x^{(i, \: 0)}_n \\
\end{pmatrix} = \begin{pmatrix} 
a^{(i, \: 0)}_1 \\
a^{(i, \: 0)}_2 \\
\cdots \\
a^{(i, \: 0)}_n \\
\end{pmatrix} = \boldsymbol{a}^{(i, \: 0)}\;,
```
then we can write a general rule for computing the outputs of a fully connected layer $\ell$ knowing the outputs of the previous layer $\ell -1$:

```{math}
:label: genrulefeedforwardeq
\boldsymbol{a}^{(i, \: \ell)} = f\left[ \; \left(W^{(\ell)}\right)^\top \; \boldsymbol{a}^{(i, \: \ell -1)} \;+\; \boldsymbol{b}^{(\ell)} \;\right]
```

This is the general rule for computing all outputs of a fully connected feedforward neural network.


## Summary
Feedforward propagation is the computation of the values of all activation units of a fully connected feedforward neural network.  

As the process includes the last layer (output), feedforward propagation also leads to predictions.  

These predictions will be compared to the observed values.  

Feedforward propagation is a step in the training of a neural network.

The next step of the training is to go 'backward', from the output error $\hat{y}^\text{pred} - y^\text{obs}$ to the first layer to then adjust all weights using a gradient-based procedure. This is the core of backpropagation, which we will cover in the next section. 


```{admonition} Learn More
:class: seealso
Very nice animations [here](https://yogayu.github.io/DeepLearningCourse/03/ForwardPropagation.html) illustrating the forward propagation process.  
Source: [Xinyu You's course _An online deep learning course for humanists_](https://yogayu.github.io/DeepLearningCourse/intro.html)
```


