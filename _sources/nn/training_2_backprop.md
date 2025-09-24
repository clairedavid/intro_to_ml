(nn:backprop)=
# Backpropagation Algorithm

## Before Diving Into The Math 
### Ingredients

The forward propagation, or forward pass, will fill the network with values for all activation units. That includes the last layer of activation units, so the forward pass provides predictions. 

We saw the loss function as the mathematical tool to compare the predictions with their associated observed values for a sample (and the cost function aggregates this for all data samples).

Then we are familiar with the gradient descent procedure, which gives at each iteration the positive or negative amount to correct the weights to eventually get a model that fits to the data.

For a neural network, there are lots of knobs to tweak! Luckily, an efficient technique called backpropagation is able to compute the gradient of the network's error for every single model parameter.

### Definition

````{prf:definition}
:label: backpropdef
__Backpropagation__, short for __backward propagation of errors__, is an algorithm working from the output nodes to the input nodes of a neural network using the chain rule to compute how much each activation unit contributed to the overall error.

It automatically computes error gradients to then repeatedly adjust all weights and biases to reduce the overall error.
````

### Chain Rule Refresher

A little interlude and refresher of the chain rule will not hurt.

````{prf:definition}
:label: chainruledef
Let $f$ and $g$ be functions. For all $𝑥$ in the domain of $g$ for which $g$ is differentiable at $x$ and $f$ is differentiable at $g(x)$, the derivative of the composite function:
\begin{equation}
h(x) = f(g(x)) 
\end{equation}
is given by
\begin{equation}
h'(x) = \frac{\mathrm{d} \; f(g(x)) }{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}} \;\cdot \;  \frac{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}}{\mathrm{d} \; x} = f'\left(g(x)\right) \;\cdot \; g'(x)
\end{equation}
````
You can see above from the colouring that the inserted denominator and numerator of the composed function cancel out. All good. Now your turn with three functions (and that will be useful for the rest of the lecture).

```{warning}
The prime notation $f'(\square)$ can be error-prone. It is the derivative with respect to $\square$ as the variable, i.e. as a block on its own (even if it depends on other variables).
\begin{equation*}
\frac{\mathrm{d} f(\square)}{\mathrm{d} \square} = f'(\square ) 
\end{equation*}
```

```{admonition} Exercise
:class: seealso
What would be the chain rule for three functions?
\begin{equation}
k(x) = h(f(g(x))) 
\end{equation}
```

````{admonition} Check your answer
:class: tip, dropdown
```{math}
:label: chainrule3funceq
\begin{align*}
k'(x) &= \; \frac{\mathrm{d} \; h(f(g(x)))}{\mathrm{{\color{BurntOrange}d}} \; {\color{BurntOrange}f(g(x))}} &\;\cdot\;& \frac{\mathrm{{\color{Peach}d}} \; {\color{Peach}f(g(x))} }{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}} &\;\cdot \;&  \frac{\mathrm{{\color{Maroon}d}} \; {\color{Maroon}g(x)}}{\mathrm{d} \; x} \\[1ex]
&= \; h'(f(g(x))) &\;\cdot\;& f'\left(g(x)\right) &\;\cdot \;& g'(x)\\
\end{align*}
```

Three functions: three derivative terms.  
We work from the outside first, taking one derivative at a time.
````



(NN2:backprop:mainstep)=
## Main Steps
Before diving into a more mathematical writing, let's just list the main steps of backpropagation. We will detail steps 2 and 3 very soon:

````{prf:algorithm} Backpropagation
:label: backpropalgo

__Inputs__  
Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $y$

__Hyperparameters__
* Learning rate $\alpha$
* Number of epochs $N$

__Start__


__Step 0:__ Weight initialization

__Step 1:__ Forward propagation:  
$\qquad \qquad \qquad \qquad \bullet$ get list of $m$ predictions $\boldsymbol{\hat{y}}^{(i)}$

__Step 2:__ Backpropagation:  
 $\qquad \qquad \qquad \qquad \bullet$ get all errors $\boldsymbol{\delta}^{(i, \: \ell)}$ using observations $\boldsymbol{y}^{(i)} \\[2ex]$  
$\qquad \qquad \qquad \qquad \bullet$ sum errors and get all cost derivatives: 
```{math}
\frac{\partial \text{ Cost}}{\partial W^{(\ell)}} \qquad ; \qquad \frac{\partial \text{ Cost}}{\partial \boldsymbol{b}^{(\ell)}}
```
__Step 3:__ Gradient Descent steps to update weights & biases:
```{math}
\begin{align*}
W^{(\ell)} &\leftarrow W^{(\ell)} - \alpha \frac{\partial \text{ Cost}}{\partial W^{(\ell)}} \\[1ex]
\boldsymbol{b}^{(\ell)} &\leftarrow \boldsymbol{b}^{(\ell)} - \alpha \frac{\partial \text{ Cost}}{\partial \boldsymbol{b}^{(\ell)}}
\end{align*}

```

End of epoch, repeat step 1 - 3 until/unless:

__Exit conditions:__
* Number of epochs $N^\text{epoch}$ is reached
* If all derivatives are zero or below a small threshold 
````

## Computations

Now there will be math.

### What is the goal?
Always a good question to start. We want to tweak the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ so that the network predictions $\boldsymbol{\hat{y}}^{(i)}$ get as close as they can be to the observed values $\boldsymbol{y}^{(i)}$. For all samples $i$ of the training dataset. In other words, we want to know how to change the weights and biases so that we minimize the cost: 
```{math}
:label: costnnmineq
 \min_{\boldsymbol{W},\boldsymbol{b}} \text{ Cost}(\boldsymbol{W},\boldsymbol{b})
```

For this, we will need the partial derivatives of the cost function with respect to the parameters we want to optimize.

With derivatives, especially partial ones, it's crucial to ask the question:

What varies here and with respect to what?  
How will the numerator entity change as the denominator change?  
Here we want to know how to vary the weights and biases so that the cost gets lower:
```{math}
:label: partialdevcostWbeq
\begin{gathered}
\frac{\partial \text{ Cost}( \boldsymbol{W},\boldsymbol{b} )}{\partial \boldsymbol{W}} \qquad \frac{\partial \text{ Cost}( \boldsymbol{W},\boldsymbol{b} )}{\partial \boldsymbol{b}}
\end{gathered}
```

```{warning}
Most mathematical textbooks have $x$ as the varying entity while explaining the derivative business. It can be here misleading with our notation as we use $\boldsymbol{x}$ as well. But in our case, $\boldsymbol{x}$ are the input features. They are given. They will not change (unless you bring new data, but there will still be given numbers you're not supposed to tweak). What we want is to vary the weights $\boldsymbol{W}$ and biases $\boldsymbol{b}$ to find optimized values for which the error is minimum, i.e. the model predicts a $\hat{y}$ as close as possible as the real target $y$.

```

Equation {eq}`partialdevcostWbeq` can be overwhelming, especially given the numerous quantities of weights and biases in a neural network. No panic! Thanks to backpropagation, there will be a way to not only get those derivatives, but also be very efficient in their computation.  


### Notations
Let's first rewrite the activation unit equation as a function of a function. We saw in the previous lecture:
```{math}
:label: activvalueeq
\boldsymbol{a}^{(i, \: \ell)} = f\left[ \; \left(W^{(\ell)}\right)^\top \; \boldsymbol{a}^{(i, \: \ell -1)} \;+\; \boldsymbol{b}^{(\ell)} \;\right]
```
with $f$ the node's activation function and $\ell$ is the current layer of the activation unit. Let's split the notation by extracting the sum:
```{math}
:label: zfunceq
\boldsymbol{z}^{(i, \: \ell)} = \left(W^{(\ell)}\right)^\top \; \boldsymbol{a}^{(i, \: \ell -1)} \;+\; \boldsymbol{b}^{(\ell)}
```

This would be called the "weighted sum plus bias." So then each activation unit can be computed as:
````{margin}
$\boldsymbol{z}^{(i, \: \ell)}$ has the same shape as $\boldsymbol{a}^{(i, \: \ell)}$.  
It is a column vector of size ($n_{\ell}$, 1), with $n_{\ell}$ the number of nodes on the layer $\ell$.
````
```{math}
:label: afzeq
\boldsymbol{a}^{(i, \: \ell)} = f \left(\boldsymbol{z}^{(i, \: \ell)} \right)
```

We will denote the loss function through a general form as $L$:
```{math}
:label: lossfunceq
L\left(\boldsymbol{\hat{y}}^{(i)}, \boldsymbol{y}^{(i)} \right) 
```
It is computed for each sample instance $\left\{ \boldsymbol{x}^{(i)}, \boldsymbol{y}^{(i)} \right\}$, with $\left(\boldsymbol{x}^{(i)}\right)^\top = (x_1^{(i)}, x_2^{(i)}, \cdots, x_n^{(i)})^\top$ being the vector of $n$ input features and $\boldsymbol{y}^{(i)}$ the associated targets.

The cost is the sum of the losses over all data instances $m$.

```{math}
:label: costfunceq
\text{Cost} = \frac{1}{m} \sum_{i=1}^m L\left(\boldsymbol{\hat{y}}^{(i)}, \boldsymbol{y}^{(i)}\right) 
```


__How can we express the final output $\boldsymbol{\hat{y}}$?__  
Let's take a similar network as the one in the previous lecture but layers are labeled from the last one (right) in decreasing order:
```{figure} ../images/training_2_nn_lastlayers.png
---
  name: training_2_3_nn_lastlayers
  width: 100%
---
 . Feedforward neural network with the notations for the  
 last, before last and before before last layers.    
<sub>Image from the author</sub>
```

The final prediction $\hat{y}^{(i)}$ is the output of the activation unit in the last layer:
```{math}
:label: ypredaL
\hat{y}_1^{(i)} = a_1^{(i, \: L)} 
```
In the network above, there is only one activation unit, so we can omit the subscript. If the network has $K$ outputs instead of one, we would have column vectors of $K$ elements:
```{math}
:label: ypredbold
\boldsymbol{\hat{y}}^{(i)} = \boldsymbol{a}^{(i, \: L)}
```

So far, so good. Now the cost.


***  
The cost function is obtained using Equations {eq}`afzeq`, {eq}`costfunceq` and {eq}`ypredbold`:
```{math}
:label: costlossafzeq
\text{Cost} = \frac{1}{m} \sum_{i=1}^m L\left(\boldsymbol{\hat{y}}^{(i)}, \boldsymbol{y}^{(i)}\right) = \frac{1}{m} \sum_{i=1}^m L\left(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)}\right) = \; \frac{1}{m} \; \sum_{i=1}^m \; L \biggl(\;f \Bigl( \;\boldsymbol{z}^{(i, \; L)}\Bigl), \boldsymbol{y}^{(i)} \biggl)
```
*** 

There are three functions here: $L \Bigl( \: f\bigl( \: z ( \:\boldsymbol{W, b} ) \bigl) \Bigl)$. 

Let's joyfully take the derivatives of that sandwich of functions! Now do you get the chain rule refresher? 

We will use it, starting with the last layer and see how things simplify (yes, it will). Then we will backpropagate layer after layer.


### The backward walk
As its name indicates, the backward propagation proceeds from the last to the first input layer. 
Let's write the derivative of the cost function with respect to the weight matrix of the last layer and apply the chain rule:
```{math}
:label: dCostlastchaineq
\begin{align*}
\frac{\partial \text { Cost }}{\partial \; W^{(L)}} &= \; \frac{1}{m} \sum_{i=1}^m \; \frac{\partial L(f(\boldsymbol{z}^{(i, \: L)}), \boldsymbol{y}^{(i)})}{\partial \; W^{(L)}} \\[2ex]
&= \; \frac{1}{m} \sum_{i=1}^m \; \frac{\partial L(f(\boldsymbol{z}^{(i, \: L)}), \boldsymbol{y}^{(i)})}{\partial f(\boldsymbol{z}^{(i, \: L)})} \; \cdot \;  \frac{\partial f(\boldsymbol{z}^{(i, \: L)})}{\partial \boldsymbol{z}^{(i, \: L)}} \; \cdot \; \frac{\partial \boldsymbol{z}^{(i, \: L)}}{\partial W^{(L)}}
\end{align*}
```


For now, the dot "$\cdot$" is ill-defined. We will write proper matrix products later. We can simplify things. The first term is the derivative of the loss function with argument $f(\boldsymbol{z}^{(i, \: L)}) = \boldsymbol{a}^{(i, \: L)}$:
```{math}
:label: dCostlprime
\frac{\partial L(f(\boldsymbol{z}^{(i, \: L)}), \boldsymbol{y}^{(i)})}{\partial f(\boldsymbol{z}^{(i, \: L)})} = L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)})
```
We have done the forward propagation, so we have access to the values for all the $\boldsymbol{a}^{(i, \: L)}$. This term is known!

Same for the second term, which is the derivative of the activation function evaluated at $\boldsymbol{z}^{(i, \: L)}$:
```{math}
:label: dCostfprime
\frac{\partial f(\boldsymbol{z}^{(i, \: L)})}{\partial \boldsymbol{z}^{(i, \: L)}} = f'(\boldsymbol{z}^{(i, \: L)})
```
This is also known!

For the third term, let's recall the definition of the $z$ function in Equation {eq}`zfunceq`.  

With a bit of math, we can show that: 
```{math}
:label: dCostdzdwlastlayer
\frac{\partial z_j^{(i, \: L)}}{\partial w_{kj}^{(L)}} = a_k^{(i, \: L-1)}
```

We actually know all these terms! 

__But how to combine them together properly?__  
Let's first consider the dimensions of this product of three derivatives. What do we want? Looking at the left hand side of equation {eq}`dCostlastchaineq`, the cost is a scalar and the weight matrix $W^{(L)}$ is of shape $n_{L-1} \times n_L$, where $n_{L-1}$ and $n_L$ are the number of nodes in the before-last and last layers respectively. Eventually, we will update each weight using the gradient descent method. So our term $\frac{\partial \text { Cost }}{\partial \; W^{(L)}}$ should be of shape $n_{L-1} \times n_L$ as well. With a bit of math, using the index notation, we can show that the product will be: 

```{math}
:label: dCostfirstvectorouterprod
\frac{\partial \text { Cost }}{\partial \; W^{(L)}} = \frac{1}{m} \sum_{i=1}^m \; \boldsymbol{a}^{(i, \: L-1)} \otimes \left[  f'(\boldsymbol{z}^{(i, \: L)}) \odot L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)}) \right]  \;,
```
where $\otimes$ is the outer product and $\odot$ denotes the element-wise multiplication between the two column vectors. You can check the outer product will be of the desired dimensions of $n_{L-1} \times n_L$.


__Now let's proceed to the before last layer.__  
Using the chain rule as usual:
```{math}
:label: dCostbeforelastchaineq
\begin{align*}
& \frac{\partial \text { Cost }}{\partial W^{(L-1)}} =  \; \frac{1}{m}\sum_{i=1}^m  \\[1ex]
& \frac{\partial L(f(\boldsymbol{z}^{(i, \: L)}), \boldsymbol{y}^{(i)})}{\partial f(\boldsymbol{z}^{(i, \: L)})} \; \cdot \;  \frac{\partial f(\boldsymbol{z}^{(i, \: L)})}{\partial \boldsymbol{z}^{(i, \: L)}} \;\cdot \; \frac{\partial \boldsymbol{z}^{(i, \: L)}}{\partial \boldsymbol{a}^{(i, \: L-1)}}  \; \cdot \; \frac{\partial \; \boldsymbol{a}^{(i, \: L-1)} }{\partial\;\boldsymbol{z}^{(i, \: L-1)} } \; \cdot \; \frac{\partial \; \boldsymbol{z}^{(i, \: L-1)} }{\partial\;W^{(L-1)} }
\end{align*}
```

The two first terms are identical as in Equation {eq}`dCostlastchaineq`. For the remaining three terms, we can show using the definitions of $\boldsymbol{a}$ and $\boldsymbol{z}$: 
```{math}
:label: beforeLasttermssimplereq
\begin{array}{c c c c c}
\displaystyle\frac{\partial \: \boldsymbol{z}^{(i, \: L)} }{\partial\ \boldsymbol{a}^{(i, \: L-1)} } & \ & \displaystyle\frac{\partial \: \boldsymbol{a}^{(i, \: L-1)} }{\partial\ \boldsymbol{z}^{(i, \: L-1)} } & \ & \displaystyle\frac{\partial \: \boldsymbol{z}^{(i, \: L-1)} }{\partial\ W^{(L-1)} }\\[2ex]
\downarrow & \ & \downarrow & \ & \downarrow\\[2ex]
\displaystyle W^{(L)} & \ &\displaystyle f'(\boldsymbol{z}^{(i, \: L-1)}) & \ & \displaystyle \boldsymbol{a}^{(i, \: L-2)}
\end{array}
```

How to properly multiply these matrix and vectors? Using again the index notation, we can reach with a bit of math this expression: 
```{math}
:label: dCostbeforelastsimpleeq
\begin{align*}
& \frac{\partial \text { Cost }}{\partial \; W^{(L-1)}} = \frac{1}{m} \sum_{i=1}^m \: \\[1ex] 
& \boldsymbol{a}^{(i, \: L-2)} \;\otimes\; \biggr[ f'(\boldsymbol{z}^{(i, \: L-1)}) \;\odot\; \Bigr[ W^{(L)} \bigr[  f'(\boldsymbol{z}^{(i, \: L)}) \: \odot \: L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)}) \bigr] \Bigr] \biggr] 
\end{align*}
```


You can check yourself that for the derivative with respect to $W^{(L-2)}$ we will have (scroll to the right, it's lengthy):
```{math}
:label: dCostbeforebeforelastsimpleeq
\begin{align*}
& \frac{\partial \text { Cost }}{\partial W^{(L-2)}} = \; \frac{1}{m} \sum_{i=1}^m \\[2ex]
&   \frac{\partial L(f(\boldsymbol{z}^{(i, \: L)}), \boldsymbol{y}^{(i)})}{\partial f(\boldsymbol{z}^{(i, \: L)})} \: \cdot \:  \frac{\partial f(\boldsymbol{z}^{(i, \: L)})}{\partial \boldsymbol{z}^{(i, \: L)}} \:\cdot \: \frac{\partial \boldsymbol{z}^{(i, \: L)}}{\partial \boldsymbol{a}^{(i, \: L-1)}}  \: \cdot \: \frac{\partial \: \boldsymbol{a}^{(i, \: L-1)} }{\partial\;\boldsymbol{z}^{(i, \: L-1)} } \: \cdot \: \frac{\partial \: \boldsymbol{z}^{(i, \: L-1)} }{\partial\ \boldsymbol{a}^{(i, \: L-2)} } \: \cdot \: \frac{\partial \: \boldsymbol{a}^{(i, \: L-2)} }{\partial\ \boldsymbol{z}^{(i, \: L-2)} } \: \cdot \: \frac{\partial \: \boldsymbol{z}^{(i, \: L-2)} }{\partial\ W^{(L-2)} }
\end{align*}
```

Here the first four terms are the same as in Equation {eq}`dCostbeforelastchaineq` but instead of the last term of Equation {eq}`dCostbeforelastchaineq`, $\boldsymbol{a}^{(i, \: L-1)}$, we have three new terms, whose derivatives are obtained using again the definition of the $z$ function in Equation {eq}`zfunceq`.
```{math}
:label: threenewtermsWminus2
\begin{array}{c c c c c}
\displaystyle\frac{\partial \: \boldsymbol{z}^{(i, \: L-1)} }{\partial\ \boldsymbol{a}^{(i, \: L-2)} } & \ & \displaystyle\frac{\partial \: \boldsymbol{a}^{(i, \: L-2)} }{\partial\ \boldsymbol{z}^{(i, \: L-2)} } & \ & \displaystyle\frac{\partial \: \boldsymbol{z}^{(i, \: L-2)} }{\partial\ W^{(L-2)} }\\[2ex]
\downarrow & \ & \downarrow & \ & \downarrow\\[2ex]
\displaystyle W^{(L-1)} & \ &\displaystyle f'(\boldsymbol{z}^{(i, \: L-2)}) & \ & \displaystyle \boldsymbol{a}^{(i, \: L-3)}
\end{array}
```
Again with the index notations (and some math), we can work out the proper operations between these vector and matrix derivatives:  
````{margin}
<br><br><br>$\leftarrow$ Scroll to the right.
````
```{math}
:label: Lminus2outerprodelementwiseall
\begin{align*}
& \frac{\partial \text { Cost }}{\partial \; W^{(L-1)}} = \frac{1}{m} \sum_{i=1}^m \: \\[1ex] 
& \boldsymbol{a}^{(i, \: L-3)} \otimes \Biggr[ f'(\boldsymbol{z}^{(i, \: L-2)}) \: \odot \: \biggr[ W^{(L-1)} \Bigr[ f'(\boldsymbol{z}^{(i, \: L-1)}) \: \odot \: \bigr[ W^{(L)} \bigr[  f'(\boldsymbol{z}^{(i, \: L)}) \: \odot \: L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)}) \bigr] \bigr] \Bigr] \biggr] \Biggr]
\end{align*}
```

Can you start to see a pattern here? Compare the equation above with {eq}`dCostbeforelastsimpleeq`. 

In the next section, we will write the terms with a much lighter notation to make this pattern more obvious. 


### Memoization (and it's not a typo)
This is a computer science term. It refers to an optimization technique to make computations faster, in particular by reusing previous calculations. This translates into storing intermediary results so that they are called again if needed, not recomputed. Recursive functions by definition reuse the outcomes of the previous iteration at the current one, so memoization is at play. 

Let's illustrate this point by writing the derivative equations for a network with three hidden layers. The output layer will be $L = 4$. Let's write the backpropagation terms.

```{warning}
In the following equations -- for this section only -- the sample index $i$ is omitted for clarity. Only the layer number is shown in the upper-script. The dot operator "$\cdot$" is unspecified at this point. We will see later the proper matrix operations.
```

```{math}
:label: lastfoursimpleeq
\begin{align*}
\frac{\partial \text { Cost }}{\partial W^{(4)}} &= \; \frac{1}{m} \; \sum \; 
{\color{OliveGreen}L^{\prime}(\boldsymbol{a}^{(4)}, \boldsymbol{y}) \cdot f^{\prime}(\boldsymbol{z}^{(4)})} \cdot \boldsymbol{a}^{(3)} \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{(3)}} &= \; \frac{1}{m} \; \sum \; 
{\color{OliveGreen}L^{\prime}(\boldsymbol{a}^{(4)}, \boldsymbol{y}) \cdot f^{\prime}(\boldsymbol{z}^{(4)})} \cdot {\color{Cyan}W^{(4)} \cdot f'(\boldsymbol{z}^{(3)})} \cdot \boldsymbol{a}^{(2)}  \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{(2)}} &= \; \frac{1}{m} \; \sum \; 
{\color{OliveGreen}L^{\prime}(\boldsymbol{a}^{(4)}, \boldsymbol{y}) \cdot f^{\prime}(\boldsymbol{z}^{(4)})} \cdot {\color{Cyan}W^{(4)} \cdot f'(\boldsymbol{z}^{(3)})} \cdot {\color{DarkOrange}W^{(3)} \cdot f'(\boldsymbol{z}^{(2)})} \cdot \boldsymbol{a}^{(1)} \\[2ex]
\frac{\partial \text { Cost }}{\partial W^{(1)}} &= \; \frac{1}{m} \; \sum \; 
{\color{OliveGreen}L^{\prime}(\boldsymbol{a}^{(4)}, \boldsymbol{y}) \cdot f^{\prime}(\boldsymbol{z}^{(4)})} \cdot {\color{Cyan}W^{(4)} \cdot f'(\boldsymbol{z}^{(3)})} \cdot {\color{DarkOrange}W^{(3)} \cdot f'(\boldsymbol{z}^{(2)})}  \cdot  {\color{awesome} W^{(2)} \cdot f'(\boldsymbol{z}^{(1)})} \cdot \boldsymbol{a}^{(0)}\\[2ex]
\end{align*}
```
The reoccuring computations are highlighted in the same colour. Now you can get a sense of the genius behind neural network: although there are many computations, a lot of calculations are reused as we move backwards through the network. With proper memoization, the whole process can be very fast. 


We will now write a general formula for backpropagation -- you may have guessed it: it will be a recursive one! 


### Recursive error equations
We can write Equations {eq}`dCostfirstvectorouterprod` and {eq}`dCostbeforelastsimpleeq` by introducing an error term $\boldsymbol{\delta}^{(i, \; \ell)}$. It will be of the same shape as the activation unit vector $\boldsymbol{a}^{(i \; \ell)}$, that is to say $n_\ell \times 1$. And at each node there will be different error values for each sample $i$.

__Error at the last layer__  
Let's define $\boldsymbol{\delta}^{(i, \; L)}$ as the product of the derivative of the loss and the activation function at the last layer (last two terms of Equation {eq}`dCostfirstvectorouterprod`): 

```{math}
:label: defdeltaL
\boldsymbol{\delta}^{(i, \; L)} \; = \; f'(\boldsymbol{z}^{(i, \: L)}) \: \odot \: L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)})
```
```{image} ../images/training_2_tetris_deltaLayerL.png
:alt: tetrominoDeltaL
:width: 90%
:align: center
```
The schematic above illustrate the element-wise product done on the right hand side, where each term is a column vector of $n_L$ elements. As we already use the left/right and up/down directions for matrix and vector operations, the sample index $i$ is here the ‘depth’, represented by several piled up sheets, aka data samples (and only four for illustration purpose).

__Derivative of the cost at the last layer__  
Looking at Equation {eq}`dCostfirstvectorouterprod`, we will have:
```{math}
:label: costWLdeltaL
\frac{\partial \text { Cost }}{\partial \; W^{(L)}} = \frac{1}{m} \sum_{i=1}^m \; \boldsymbol{a}^{(i, \: L-1)} \otimes \left[ \: \boldsymbol{\delta}^{(i, \; L)} \: \right]
```
In terms of dimensions, the outer product creates a $n_{L-1} \times n_L$ matrix; this is what we want to get the correct dimensionality on the left hand side:

```{image} ../images/training_2_tetris_dCostdWlastLayer.png
:alt: tetris_dCostdWlastLayer
:width: 90%
:align: center
```
Note that the derivatives of the cost on the left hand side -- shown in green -- are the result of the summation over the $m$ samples so there is no 'depth' anymore. 


__Error at the before-last layer__  
From Equation {eq}`dCostbeforelastsimpleeq`, we inject the definition of $\boldsymbol{\delta}^{(i, \; L)}$ and define the error at the before-last layer by adding the terms:
```{math}
:label: deltaLminus1
\boldsymbol{\delta}^{(i, \; L-1)} = f'(\boldsymbol{z}^{(i, \: L-1)}) \:\odot\: \Bigr[ \: W^{(L)} \; \boldsymbol{\delta}^{(i, \; L)} \:  \Bigr]
```
Dimension-wise, the matrix multiplication $W^{(L)} \; \boldsymbol{\delta}^{(i, \; L)}$ leads to a column vector of $n_{L-1}$ elements, which is then multiplied element-wise to give a column vector of, again, $n_{L-1}$ elements.
```{image} ../images/training_2_tetris_deltaLayerLminus1.png
:alt: tetris_deltaLayerLminus1
:width: 90%
:align: center
```

__Derivative of the cost at the before-last layer__  
Moving on backward, we still use Equation {eq}`dCostbeforelastsimpleeq` to express the derivative of the cost with respect to the weights of the before-last layer the following way:
```{math}
:label: costWLdeltaLminus1
\frac{\partial \text { Cost }}{\partial \; W^{(L-1)}} = \frac{1}{m} \sum_{i=1}^m \; \boldsymbol{a}^{(i, \: L-2)} \otimes \left[ \: \boldsymbol{\delta}^{(i, \; L-1)} \: \right]
```
And the outer product gives the proper dimensions for the left hand size, i.e. a $n_{L-2} \times n_{L-1}$ matrix.
```{image} ../images/training_2_tetris_dCostdWlastlastLayer.png
:alt: training_2_tetris_dCostdWlastlastLayer
:width: 90%
:align: center
```
&nbsp;  

__General equations for errors and costs__  
We can generalize Equations {eq}`deltaLminus1` and {eq}`costWLdeltaLminus1`.  

***

Error at layer $\ell$:
```{math}
:label: deltalayerl
\begin{align*}
\boldsymbol{\delta}^{(i, \; \ell)} = f'(\boldsymbol{z}^{(i, \: \ell)}) \:\odot\: \Bigr[ \: W^{(\ell + 1)} \; \boldsymbol{\delta}^{(i, \; \ell + 1)} \:  \Bigr] 
\end{align*}
```


Derivative of the cost at layer $\ell$:
```{math}
:label: dCostlayerl
\begin{align*}
\frac{\partial \text { Cost }}{\partial \; W^{(\ell)}} = \frac{1}{m} \sum_{i=1}^m \; \boldsymbol{a}^{(i, \: \ell -1)} \otimes \left[ \: \boldsymbol{\delta}^{(i, \; \ell)} \: \right]
\end{align*}
```

***

This has become much simpler, hasn't it?


__What about the biases?__  
This is left as exercise for training. 
```{admonition} Exercise
:class: seealso
Express the partial derivatives of the cost with respect to the biases $\boldsymbol{b}^{(\ell)}$.

Hint: start with the last layer $L$ as done previously with the weights.
```

````{admonition} Check your answer
:class: tip, dropdown
See the Summary section below.
````



### Weights and biases update
After backpropagating, each weight and each bias in the network are adjusted in proportion to how much they contribute to the overall error.  

```{math}
:label: weightbiasupdate
\begin{align*}
&W^{(\ell)} &\leftarrow& &W^{(\ell)} \quad &-& \alpha \frac{\partial \text{ Cost}}{\partial W^{(\ell)}} \\[2ex]
&\boldsymbol{b}^{(\ell)} &\leftarrow& &\boldsymbol{b}^{(\ell)} \quad &-& \alpha \frac{\partial \text{ Cost}}{\partial \boldsymbol{b}^{(\ell)}}
\end{align*}
```
Nothing new here, it is the standard gradient descent step we are familiar with. There are just more elements to update. 

```{admonition} Exercise
:class: seealso
For a network of $L$ layers, that is to say $L -1$ hidden layers plus an output layer, how many weight matrices $W^{(\ell)}$ and how many biases vectors $\boldsymbol{b}^{(\ell)}$ do we have to update?
```
Let's now wrap it up!

## Summary on backpropagation
The backpropagation of error is a recursive algorithm reusing the computations from last until first layer to compute how much each activation unit and bias node contribute to the overall error. The idea behind backpropagation is to share the repeated computations wherever possible. 
Let's write again the steps with the key equations:

````{prf:algorithm} Backpropagation
:label: backpropalgosummary

__Inputs__  
Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $\boldsymbol{y}$

__Hyperparameters__
* Learning rate $\alpha$
* Number of epochs $N$

__Start__


__Step 0: Weight initialization__

__Step 1: Forward propagation__  
$\qquad \quad \bullet$ get list of $m$ predictions $\boldsymbol{\hat{y}}^{(i)}$
\begin{equation}
 f(\boldsymbol{z}^{(i, \; L)}) = \boldsymbol{a}^{(i, \;L)} = \boldsymbol{\hat{y}}^{(i)}
\end{equation}

__Step 2: Backpropagation__  
$\qquad \quad \bullet$ get the cost:
\begin{equation}
\text{Cost} =  \; \frac{1}{m} \; \sum_{i=1}^m \; L \biggl(\;f \Bigl( \;\boldsymbol{z}^{(i, \; L)}\Bigl), \boldsymbol{y}^{(i)} \biggl)\end{equation}
$\qquad \quad \bullet$ get all errors:
```{math}
:label: summarybackproperrors
\begin{align*}
\boldsymbol{\delta}^{(i, \; L)} \; &= \quad f'(\boldsymbol{z}^{(i, \: L)}) \: &\odot& \quad L^{\prime}(\boldsymbol{a}^{(i, \: L)}, \boldsymbol{y}^{(i)}) \\[2ex]
\boldsymbol{\delta}^{(i, \; \ell)} \;  &=  \quad f'(\boldsymbol{z}^{(i, \: \ell)}) \: &\odot& \quad \Bigr[ \: W^{(\ell + 1)} \; \boldsymbol{\delta}^{(i, \; \ell + 1)} \:  \Bigr] 
\end{align*}
```

$\qquad \quad \bullet$ sum errors and get all cost derivatives:  
```{math}
:label: summarybackpropcostd
\begin{align*}
\frac{\partial \text { Cost }}{\partial \; W^{(\ell)}} &= \frac{1}{m} \sum_{i=1}^m \; \boldsymbol{a}^{(i, \: \ell -1)} \otimes \left[ \: \boldsymbol{\delta}^{(i, \; \ell)} \: \right]
\\[2ex]
\frac{\partial \text { Cost }}{\partial \; \boldsymbol{b}^{(\ell)}} &= \frac{1}{m} \sum_{i=1}^m \;  \boldsymbol{\delta}^{(i, \; \ell)}\end{align*}
```

__Step 3: Gradient Descent__  
$\qquad \quad \bullet$ update weights & biases:
```{math}
\begin{align*}
&W^{(\ell)} &\leftarrow& &W^{(\ell)} \quad &-& \alpha \frac{\partial \text{ Cost}}{\partial W^{(\ell)}} \\[2ex]
&\boldsymbol{b}^{(\ell)} &\leftarrow& &\boldsymbol{b}^{(\ell)} \quad &-& \alpha \frac{\partial \text{ Cost}}{\partial \boldsymbol{b}^{(\ell)}}
\end{align*}
```


End of epoch, repeat step 1 - 3 until/unless:

__Exit conditions:__
* Number of epochs $N^\text{epoch}$ is reached
* If all derivatives are zero or below a small threshold 
````


Now you know how neural networks are trained! 


In the assignment, you will code yourself a small neural network from scratch. Don't worry: it will be guided. In the next lecture, we will see a much more convenient way to build a neural network using dedicated libraries. We will introduce further optimization techniques specific to deep learning.


```{admonition} Exercise
:class: seealso
Now that you know the backpropagation algorithm, a question regarding the neural network initialization: what if all weights are first set to the same value? (not zero, as we saw, but any other constant)
```
````{admonition} Check your answer
:class: tip, dropdown
If the weights and biases are initialized to the same constant values $w$ and $b$, all activation units in a given layer will get the same signal $a = f(\sum_{j} w_j \; x_j + b)$. As such, all nodes for that layer will be identical. Thus the gradients will be updated the same way. Despite having many neurons per layer, the network will act as if it had only one neuron per layer. Therefore, it is likely to fail to reproduce complex patterns from the data; it won't be that smart. For a feedforward neural network to work, there should be an asymmetric configuration for it to use each activation unit uniquely. This is why weights and biases should be initalized with random value to break the symmetry.
````


&nbsp;&nbsp;


```{admonition} Learn more
:class: seealso
The paper that popularized backpropagation, back in 1989:  
[D. Rumelhart, G. Hinton and R.Williams, _Learning representations by back-propagating errors_](https://www.nature.com/articles/323533a0)

Some good refresher:  
[Derivative of the chain rule on math.libretexts.org](https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/03%3A_Derivatives/3.06%3A_The_Chain_Rule)

Backpropagation explanation with different notation and a source of inspiration (thanks) from [ml-cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html)

```