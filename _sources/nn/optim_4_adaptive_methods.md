# Adaptive Optimizers

Before the arrival of faster methods, combining Stochastic Gradient Descent with a learning rate schedule was considered close to state-of-the-art. Adaptive optimizers changed this by embedding the learning rate adjustment directly into the optimization process. Instead of relying on a separate scheduler, they use feedback from the model to adjust updates based on past gradients. The key advantage: they require little to no manual tuning.

````{prf:definition}
:label: 
An __Adaptative Learning Rate__ is a technique that varies the learning rate using feedback from the model itself.
````
As the variation of the learning rate is done by the optimizer, the terms adaptive learning rate and adaptive optimizer are often used interchangeably.

Below are brief descriptions of the most popular adaptative optimizers.


## AdaGrad
This first adaptative algorithm was published in 2011. Compared to the classical Gradient Descent, AdaGrad points more directly toward the global optimum by decaying the learning rate more on the steepest dimension. 

```{figure} ../images/optim_4_adagrad.png
---
  name: optim_4_adagrad
  width: 80%
---
 . The Gradient Descent (blue) vs AdaGrad (orange).  
<sub>Image: Aurélien Géron, _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, Second Edition_</sub> 
```
In the figure above, the AdaGrad takes a more direct thus shorter path than Gradient Descent, as AdaGrad has its learning rate reduced in the direction of steepest descent. This is done by squaring each gradient component into a vector $\boldsymbol{s}$. The steepest a gradient component, the larger the square of this component $s_j$. The weights are updated in almost the same way as with Gradient Descent, yet each component is divided by $\boldsymbol{s} + \epsilon$. The added term $\epsilon$ is to prevent a division by zero (typically 10$^{-10}$). As a result, the algorithm detects how to change the learning rate dynamically and specifically on the large gradient components to adapt to their steep slope.

Mathematically: 
````{margin}
The $\otimes$ symbol is the element-wise multiplication, while the $\oslash$ is the element-wise division.
````
```{math}
\begin{align*}
\boldsymbol{s} \; &\leftarrow \; \boldsymbol{s}  \;+\;  \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \;\otimes \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \\[2ex]
\boldsymbol{W} \; &\leftarrow \; \boldsymbol{W} - \alpha \; \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \oslash \sqrt{\boldsymbol{s} + \epsilon}
\end{align*}
```
__The con__  
AdaGrad performs well on simple problem, like linear regression. With neural networks, it tends to stop too soon, before reaching the global minimum. Luckily, other adaptative algorithms fix this.

## RMSProp
RMSProp stands for Root Mean Square Propagation. It works the same as AdaGrad except that it keeps track of an exponentially decaying average of past squared gradients: 
```{math}
:label: rmspropeq
\begin{align*}
\boldsymbol{s} \; &\leftarrow \; \beta\boldsymbol{s}  \;+\;  (1 - \beta) \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \;\otimes \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \\[2ex]
\boldsymbol{W} \; &\leftarrow \; \boldsymbol{W} - \alpha \; \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \oslash \sqrt{\boldsymbol{s} + \epsilon}
\end{align*}
```
with the decay rate $\beta$, between 0 and 1, yet typically set to 0.9. One might ask: what is the point to introduce an extra hyparameter? It turns out the default value works well in most cases and does not require tuning. What the update in the expressions {eq}`rmspropeq` shows is a recursive way of computing a so-called Exponential Moving Average (EMA). In other words, $\beta$ acts as a constant smoothing factor, representing the degree of weighting increase. A lower $\beta$ discounts older observations faster. A higher $\beta$ gives more weight to the previous gradient, a bit less weight to the previous previous gradient, etc.

For more elaborated tasks than the simple linear case, RMSProp is robust. It reigned until dethrowned by a newcomer called Adam. Before introducing Adam, let's first cover the notion of momentum optimization.

## Momentum Optimization
````{margin}
```{warning}
The algorithm name borrows the momentum concept from physics, yet it is a only metaphor. 
```
````
In physics, the momentum $\boldsymbol{p}$ is a vector obtained by taking the product of the mass and velocity of an object. It quantifies motion. In computing science, momentum refers to the direction and speed at which the parameters move - via iterative updates - through the parameter space. With momentum optimization, inertia is added to the system by updating the weights using the momenta from past iterations. This keeps the update in the same direction. The common analogy is a ball rolling down on a curved surface. It will start slowly but soon "gain momentum", thus can go through flat gradient surface much faster than with the classic Gradient Descent. Adding momentum considerably speeds the Gradient Descent process. It also helps roll beside local minimum. 

The momentum optimization algorithm is as follow:
```{math}
\begin{align*}
\boldsymbol{m} \; &\leftarrow \; \beta \boldsymbol{m} \; - \; \alpha \; \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \\[2ex]
\boldsymbol{W} \; &\leftarrow \; \boldsymbol{W} + \boldsymbol{m} 
\end{align*}
```
Here the $\beta$ parameter controls the momentum from becoming too large. It could be analogous to introducing friction; $\beta$ ranges from 0 to 1, with 0 meaning high friction and 1 no friction at all. In the literature you will see a common default value of $\beta$ = 0.9.

## Adam
Adam stands for _Adaptative moment estimation_. It merges RMSProp with momentum optimization. Recall that RMSProp uses an exponentially decaying average of past squared gradients, while momentum does the same except with gradients (not squared). 

Mathematically:
```{math}
:label: adameq
\begin{align*}
\boldsymbol{m} \; &\leftarrow \; \beta_1 \boldsymbol{m} \; - \; (1 - \beta_1) \; \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}}\\[1ex]
\boldsymbol{s} \; &\leftarrow \; \; \beta_2\boldsymbol{s}  \;+\;  (1 - \beta_2) \; \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \otimes  \frac{\partial J\left(\boldsymbol{W}\right)}{\partial \boldsymbol{W}} \\[1ex]
\boldsymbol{\widehat{m}} &\leftarrow \frac{\boldsymbol{m}}{1 - \beta_1^t} \\[1ex]
\boldsymbol{\widehat{s}} &\leftarrow \frac{\boldsymbol{s}}{1 - \beta_2^t} \\[1ex]
\boldsymbol{W} \; &\leftarrow \; \boldsymbol{W} + \alpha \; \boldsymbol{\widehat{m}} \oslash \sqrt{\boldsymbol{\widehat{s}} + \epsilon}
\end{align*} 
```
with $t$ the iteration number.

The first step is not exactly the momentum; instead of an exponentially decaying sum, Adam computes an exponentially decaying average. 

The algorithm seems complicated at first glance, especially steps including the averages of $\boldsymbol{\widehat{m}}$ and $\boldsymbol{\widehat{s}}$. The division with the betas is to boost at the start of the training the values or $\boldsymbol{m}$ and $\boldsymbol{s}$, which are initialized at zero, in order to not stay at zero.

"Many extra parameters on top of the learning rate, so what is the point?" will you say. Actually the momentum decay $\beta_1$ is set to 0.9, the scaling one $\beta_2$ to 0.999 and $\epsilon$ to 10$^{-7}$. And if the learning rate is still a hyperparameter (usually set at 0.001 at start), it will be adapted to the optimiziation problem at hand by the algorithm. In that sense, the Adam optimizer is almost parameter free.

```{admonition} Learn More
:class: seealso

* Article: [How to Configure the Learning Rate When Training Deep Learning Neural Networks](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/) (machinelearningmastery.com)  
* Article: [Gradient Descent With Momentum from Scratch](https://machinelearningmastery.com/gradient-descent-with-momentum-from-scratch/) (machinelearningmastery.com)  
* Paper: [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html) (JMLR)  
* There is no paper for RMSProp as it was never officially published! One of the authors Geoffrey Hinton presented it in a Coursera lecture. As a consequence, it is amusingly referenced by researchers as: [Slide 29 in Lecture 6](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) (Hinton, Coursera, 2012)
* Article: [Gradient Descent with Momentum](https://towardsdatascience.com/gradient-descent-with-momentum-59420f626c8f) (towardsdatascience.com)  
👉 helps understand the Exponential Moving Average (EMA).  
* Paper: [Adam: A Method for Stochastic Optimization (2015)](https://arxiv.org/abs/1412.6980) (arXiv)  

```