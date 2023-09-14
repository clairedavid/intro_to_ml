# Feedforward Propagation


## What is Feedforward Propagation?

It is a first step in the training of a neural network (after initialization of the weights, which will be covered in the next lecture). The forward direction means going from input to output nodes. 

````{prf:definition}
:label: feedforwardpropdef
The __Feedforward Propagation__, also called __Forward Pass__, is the process consisting of computing and storing all network nodes' output values, starting with the first hidden layer until the last output layer, using at start either a subset or the entire dataset samples.
````

Forward propagation thus leads to a list of the neural network predictions for each data instance row used as input. At each node, the computation is the key equation {eq}`aneq` we saw in the previous Section {ref}`NN1:modelRep`, written again for convenience:
```{math}
:label: sumwixieq
y = f\left(\sum_{j=1}^n w_j x_j + b \right)
```

But there will be some change in the notations. Let's define everything in the next subsection.

## Notations
Let's say we have the following network with $x_n$ input features, one first hidden layer with $q$ activation units and a second one with $r$ activation units. For simplicity, we will choose an output layer with only one node:


```{figure} ../images/lec05_4_nn_notations.png
---
  name: lec05_4_nn_notations
  width: 100%
---
 . A feedforward neural network with the notation we will use for the forward propagation equations (more in text).    
<sub>Image from the author</sub>
```

There are lots of subscripts and upperscripts here. Let's explain the conventions we will use.  

__Input data__  
We saw in Lecture 2 that the dataset in supervised learning can be represented as a matrix $X$ of $m$ data instances (rows) of $n$ input features (columns). For clarity in the notations, we will focus for now on only one data instance, the $i$th sample row $\boldsymbol{x^{(i)}} = ( x_1, x_2, \cdots, x_n)$. And we will omit the $(i)$ upperscript for now.  
````{margin}
The layer numbering starts at the first hidden layer where $\ell=1$. The input layer is $\ell=0$.
````
__Activation units__  
In a given layer $\ell = 1, 2, \cdots, N^\text{layer}$, the activation units will give outputs that we will note as a row vector 
```{math}
\boldsymbol{a^{(\ell)}} = \left( a_1^{(\ell)}, a_2^{(\ell)}, \cdots , a_q^{(\ell)} \right) \;, 
```
where the upperscript is the layer number and the subscript is the row of the activation unit in the layer, starting from the top.

__Biases__  
The biases are also row vectors, one for each layer it connects to and of dimension the number of nodes in that layer:
```{math}
\boldsymbol{b^{(\ell)}} = \left( b_1^{(\ell)}, b_2^{(\ell)}, \cdots , b_q^{(\ell)}\right)
```
If the last layer is only made of one node like in our example above, then $b$ is a scalar. 

__Weights__  
Now the weights. You may see in the literature different ways to represent them. In here we use a convention we could write as:
```{math}
w^\ell_{(\ell -1) \; \to \; \ell}
```
In other words, the first index is the row of the node from the previous layer (departing node of the weight's arrow) and the second index is the row of the node from the current layer (the one the weight's arrow points to). For instance $w^{(2)}_{3,1}$ is the weight from the third node of layer (1) going to the first node of layer (2). 

We can actually represent each weight from layer $\ell -1$ to layer $\ell$ as a matrix $W^{(\ell)}$. If the previous layer $\ell -1$ has $n$ nodes and the layer $\ell$ has $q$ activation units, we will have:

```{math}
:label: Wmatrixeq
W^{(\ell)} = \begin{pmatrix}
w_{1,1}^{(\ell)} & w_{1,2}^{(\ell)} & \cdots & w_{1,q}^{(\ell)} \\[2ex]
w_{2,1}^{(\ell)} & w_{2,2}^{(\ell)} & \cdots & w_{2,q}^{(\ell)} \\[1ex]
\vdots  & \vdots & \ddots   & \vdots \\[1ex]
w_{n,1}^{(\ell)} & w_{n,2}^{(\ell)} &  \cdots & w_{n,q}^{(\ell)} \\
\end{pmatrix}
```

Let's now see how we calculate all the values of the activation units!


## Step by step calculations

### Computation of the first hidden layer 
Let's use Equation {eq}`sumwixieq` to compute the activation unit outputs of the first layer. The activation function is represented as $f$ here:
```{math}
:label: firstlayereq
\begin{align*}
a^{(1)}_1 &= f\left(\; w_{1,1}^{(1)} \; x_1 \;+\; w_{2,1}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,1}^{(1)} \; x_n \;+\; b^{(1)}_1\right)\\[2ex]
a^{(1)}_2 &= f\left(\; w_{1,2}^{(1)} \; x_1 \;+\; w_{2,2}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,2}^{(1)} \; x_n \;+\; b^{(1)}_2\right)\\
&\vdots \\[2ex]
a^{(1)}_q &= f\left(\; w_{1,q}^{(1)} \; x_1 \;+\; w_{2,q}^{(1)} \; x_2 \;+\; \cdots + \; w_{n,q}^{(1)} \; x_n \;+\; b^{(1)}_q\right)\\
\end{align*}
```

  
We can actually write it in the matrix form. Let's first write it in an expanded version with the matrix elements:
```{math}
:label: firstlayermatrixexpandedeq
\boldsymbol{a^{(1)}} = f\left[ \; ( x_1, x_2, \cdots, x_n) 
\begin{pmatrix}w_{1,1}^{(1)} & w_{1,2}^{(1)} & \cdots & w_{1,q}^{(1)} \\[2ex]w_{2,1}^{(1)} & w_{2,2}^{(1)} & \cdots & w_{2,q}^{(1)} \\[1ex]\vdots  & \vdots & \ddots   & \vdots \\[1ex]w_{n,1}^{(1)} & w_{n,2}^{(1)} &  \cdots & w_{n,q}^{(1)} \\\end{pmatrix}
 \;+\; ( b_1^{(1)}, b_2^{(1)}, \cdots , b_q^{(1)}) \; \right]
```

This can be written in a compact way:
```{math}
:label: firstlayermatrixeq
\boldsymbol{a^{(1)}} = f\left( \; \boldsymbol{x} \;W^{(1)} \;+\; \boldsymbol{b}^{(1)} \;\right)
```
Much lighter. 

### Computation of the second hidden layer 
Let's do the same calculation for the second layer of activation units. Instead of the dataset vector $\boldsymbol{x}$, we will have $\boldsymbol{a^{(1)}}$ as input:
```{math}
:label: secondlayermatrixexpandedeq
\boldsymbol{a^{(2)}} = f\left[ \; ( a^{(1)}_1, a^{(1)}_2, \cdots, a^{(1)}_q) 
\begin{pmatrix}w_{1,1}^{(2)} & w_{1,2}^{(2)} & \cdots & w_{1,r}^{(2)} \\[2ex]
w_{2,1}^{(2)} & w_{2,2}^{(2)} & \cdots & w_{2,r}^{(2)} \\[1ex]
\vdots  & \vdots & \ddots   & \vdots \\[1ex]
w_{q,1}^{(2)} & w_{q,2}^{(2)} &  \cdots & w_{q,r}^{(2)} \\
\end{pmatrix} \;+\; ( b_1^{(2)}, b_2^{(2)}, \cdots , b_r^{(2)}) \; \right]
```

And the elegant, light version:
```{math}
:label: secondlayermatrixeq
\boldsymbol{a^{(2)}} = f\left( \; \boldsymbol{a^{(1)}} \;W^{(2)} \;+\; \boldsymbol{b}^{(2)} \;\right)
```

We start seeing a pattern here thanks to the matricial form. More on this soon in Section {ref}`NN1:forwardprop:rule`. Let's finish the process with the last layer:

### Computation of the third hidder layer 
With one output node, it is actually simpler than for the hidden layers above. We can still write it in the same form as Equation {eq}`secondlayermatrixeq`:
```{math}
:label: thirdlayermatrixeq
\boldsymbol{a^{(3)}} = f\left( \; \boldsymbol{a^{(2)}} \;W^{(3)} \;+\; \boldsymbol{b}^{(3)} \;\right)
```
with $\boldsymbol{a^{(2)}} = (a^{(2)}_1, a^{(2)}_2, \cdots, a^{(2)}_r)$ that we calculated above. 

In our case $\boldsymbol{a^{(3)}}$ has only one element: $a^{(3)}_1 = y$.

The matrix $W^{(3)}$ has only one column.

The bias 'vector' is actually a scalar: $b^{(3)}$. 

That's the end of the forward propagation process! As you can see, it contains lots of calculations. And now you may understand why activation functions that are simple and fast to compute are preferrable, as they intervene each time we compute the output of an activation unit.

Let's now generalize this with a general formula.


(NN1:forwardprop:rule)=
## General rule for Forward Propagation

If we rewrite the first layer of inputs as:
```{math}
:label: xisazeroeq
\boldsymbol{x^{(i)}} = ( x_1, x_2, \cdots, x_n) = ( a^{(0)}_1, a^{(0)}_2, \cdots, a^{(0)}_n) = \boldsymbol{a^{(0)}} \;,
```
then we can write a general rule for computing the outputs of a fully connected layer $\ell$ knowing the outputs of the previous layer $\ell$ (which become the layer $\ell$'s inputs):

```{math}
:label: genrulefeedforwardeq
\boldsymbol{a^{(\ell)}} = f\left( \; \boldsymbol{a^{(\ell -1)}} \;W^{(\ell)} \;+\; \boldsymbol{b}^{(\ell)} \;\right)
```

This is the general rule for computing all outputs of a fully connected feedforward neural network.

## Summary
Feedforward propagation is the computation of the values of all activation units of a fully connected feedforward neural network.  

As the process includes the last layer (output), feedforward propagation also leads to predictions.  

These predictions will be compared to the observed values.  

Feedforward propagation is a step in the training of a neural network.

The next step of the training is to go 'backward', from the output error $\hat{y}^\text{pred} - y^\text{obs}$ to the first layer to then adjust all weights using a gradient-based procedure. This is the core of backpropagation, which we will cover at the next lecture. 


```{admonition} Learn More
:class: seealso
Very nice animations [here](https://yogayu.github.io/DeepLearningCourse/03/ForwardPropagation.html) illustrating the forward propagation process.  
Source: [Xinyu You's course _An online deep learning course for humanists_](https://yogayu.github.io/DeepLearningCourse/intro.html)
```
