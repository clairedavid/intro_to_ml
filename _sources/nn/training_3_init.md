# Initialization Schemes

<br>

<br>

<br>

Let's start this section with the main question it will answer. 

Refrain from scrolling down too much as it contains spoilers!  

Think of it first!

<br>

<br>

<br>

```{admonition} Exercise
:class: seealso

How should the weights be initialized for a neural network to work?

Give it a try yourself and share your findings in groups of 3 or 4.
```

<br>

````{admonition} Expand if you need tips and hints
:class: tip, dropdown

Ask yourself:
* Should all weights be initialized or not?
* Would some values pose problem?
* Recall how we did for linear and logistic regression. Could we proceed the same here?
````

<br>

<br>

<br>

<br>

Initialization is a crucial processs for neural networks. The initial values can determine if the training will succeed or not, i.e. if the fit to the data will converge or not.

## Introduction
### Everyone at zero?
We could ask ourself: should we simply put all weights to zero?  
This corresponds to deactivating all neurons. The formula $\boldsymbol{a}^{(i, \: \ell)} = f\left[ \; \left(W^{(\ell)}\right)^\top \; \boldsymbol{a}^{(i, \: \ell -1)} \;+\; \boldsymbol{b}^{(\ell)} \;\right]$ would make no progress through all layers if all weights and biases are zero. 

We will see after covering backpropagation in the next section that any other constant for the weight initialization will lead to a problem. 

### Don't die, don't explode, don't saturate
These should be the rules for weight initialization. Explanations.  

If we use a different number than zero, then we will validate the "don't die" part. But which values to choose? 

````{margin}
[Xavier Glorot, Yoshua Bengio, _Understanding the difficulty of training deep feedforward neural networks_ (2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
````
The non-zero values should not lead to saturation nor explosion of the gradients. This must remind you of something. Yes: those were issues posed by some activation functions when their input values become large (Section {ref}`NN1:activationF:risksGradient`). But it has been found that the initialization could also make a neural network unstable. The whole instability problem of neural network was not clear until a paper (link on the right) in 2010 from Xavier Glorot and Yoshua Bengio revealed the culprits.

At that time, there were heuristics found before 2010 (you will see why soon) consisting of initializing all weights to random values between -1 to 1, or -0.3 to 0.3 or either asymmetrically from 0 to 1. The weight values would follow either a Gaussian or uniform distribution. These heuristics work well in general. But the paper authors showed that the combination of a sigmoid as activation function with such initialization methods produces outputs with a larger variance than the inputs at the previous layer. The phenomenon amplifies itself layer after layer, eventually leading to saturation at the last layer. And as the sigmoid is not centered on zero but 0.5, this actually makes it even worse.

Luckily the same authors in the same paper suggested a way to mitigate this, and with other experts (and papers) they proposed 'modern' initialization methods that we will present in the next section.


## Standard Initialization Strategies
Let's first define the notion of _fan-in_ and _fan-out_. 

````{prf:definition}
:label: fanindef

In a neural network, __fan-in__ refers to the number of incoming network connections  
(input neurons and bias of the previous layer)

The __fan-out__ is the number of outgoing connections, i.e. the number of neurons of the layer.

````
(NN2_init:Xavier)=
### Xavier Weight Initialization
In their 2010 paper, the authors Glorot and Bengio showed that the signal needs to flow properly while making predictions (forward propagation) as well as in the reverse direction while backpropaging the gradients. What is meant by 'flowing properly is to have a constraint on the variance:

```{prf:property}
:label: varianceinitxavier
The variance of the outputs at each layer should be equal to the variance of its inputs.
```
```{prf:property}
:label: gradientsinitxavier
The gradients should have equal variance before and after flowing through a layer in the reverse direction.
```
It is impossible to validate the two properties at the same time. But Glorot and Bengio offered a compromise that passed the experimental tests with success, mostly with the sigmoid activation function.

````{prf:definition}
:label: xavieruniformdef
The __Uniform Xavier Initialization__ is obtained by drawing each weight from a random uniform distribution in in $[-x,x]$, with 
```{math}
 x= \sqrt{\frac{6}{\textit{fan}_\text{ in} + \textit{fan}_\text{ out}}}
```
````

````{prf:definition}
:label: xaviernormaldef
The __Normal Xavier Initialization__ is obtained by drawing each weight from a random normal distribution with mean of 0, and a standard deviation $\sigma$ of:
```{math}
 \sigma = \sqrt{\frac{2}{\textit{fan}_\text{ in} + \textit{fan}_\text{ out}}}
```
````
The Xavier initialization works well for sigmoid, tanh and softmax activation functions.

What is important here is the bounding of the variance using the numbers of inputs and outputs. It has been demonstrated that the resulting, properly scaled, distribution of the weights speeds the training considerably. 


### LeCun Weight Initialization

A variant of Xavier initilization has been proposed by Yann LeCun, one of the godfathers of deep learning and among the top most influential AI researchers in the world. His trick is to only use the number of inputs, _fan-in_. 

````{prf:definition}
:label: lecuninitdef
The __LeCun Weight Initialization__ is obtained by drawing each weight from a random normal distribution with mean of 0, and a standard deviation $\sigma$ of:
```{math}
 \sigma = \sqrt{\frac{1}{\textit{fan}_\text{ in} }}
```
````
This initialization works with the SELU activation function (see Section {ref}`NN1:activationF:SELU`).

&nbsp;


Both Xavier and LeCun methods are for differentiable activation functions. The following method is for non-differentiable activation functions like ReLU.

### He Weight Initialization

Kaiming He and others published a paper providing a good initialization strategy for the ReLU activation function and its variants. 

````{prf:definition}
:label: heinitdef
The __He Weight Initialization__ is obtained by drawing each weight from a random normal distribution with mean of 0, and a standard deviation $\sigma$ of:
```{math}
 \sigma = \sqrt{\frac{2}{\textit{fan}_\text{ in} }}
```
````



```{admonition} Learn more
:class: seealso
* Interactive visualization of the effects of different initializations: [Initializing neural networks](https://www.deeplearning.ai/ai-notes/initialization/index.html) (from www.deeplearning.ai)
* Articles [Weight Initialization for Deep Learning Neural Networks](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) and [Why Initialize a Neural Network with Random Weights?](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) by Jason Brownlee (machinelearningmastery.com)
* [What is Xavier initialization?](https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/) (365datascience.com)
```
