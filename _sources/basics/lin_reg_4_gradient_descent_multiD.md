# Multivariate Linear Regression

In machine learning, data samples contain multiple features. 

We will generalize our previous definitions of linear regression to additional features. How to see this? 
Previously our data consisted of one column vector $\boldsymbol{x}$ and its associated column vector of targets $\boldsymbol{y}$.
Additional input features can be visualized in the form of a table:

```{glue:figure} df_example
:figwidth: 300px
:name: "tbl:df_example"
```
The data set can be represented as a 2D array with training examples listed as rows. Here each training example has four input features $x_1$, $x_2$, $x_3$ and $x_4$. Each column lists a given feature.


In particle physics, input features can be the properties of the particles in a collision event. For instance $x_1$ could be the electron momentum, $x_2$ the electron's azimutal angle of its direction in the detector's coordinate system, $x_3$ could be another particle's momentum, $x_4$ could be a special angle between two particles.

Again let's have consistent notations:

```{admonition} Terminology and Notation 

* The number of training examples is denoted with $m$
* The number of features is denoted with $n$  
<br>

* The row index or rank of a training example is in superscript, from 1 to $m$
* The feature or column of a training example is in subscript, from 1 to $n$  


$\Rightarrow$ the value of input feature $j$ of training example in row $i$ is
\begin{equation*}
x_j^{(i)}
\end{equation*}
The training examples form a ($m \times n$) matrix $X$:
\begin{equation*}
X = \begin{pmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots  & x_j^{(1)} & \cdots & x_n^{(1)} \\[2ex]
x_1^{(2)} & x_2^{(2)} & \cdots & x_j^{(2)} & \cdots & x_n^{(2)} \\
\vdots  & \vdots & \ddots  & \vdots &  & \vdots \\
x_1^{(i)} & x_2^{(i)} & \cdots & x_j^{(i)} & \cdots & x_n^{(i)} \\
\vdots & \vdots &  & \vdots & \ddots  & \vdots \\
x_1^{(m)} & x_2^{(m)} & \cdots & x_j^{(m)} & \cdots & x_n^{(m)} \\
\end{pmatrix} 
\end{equation*}
```

```{warning}
The $i^\text{th}$ training sample $\boldsymbol{x^{(i)}}$ is not a scalar but a row vector of $n$ elements.  
```

Our hypothesis function is generalized to the following for a given training example (row $i$):
````{prf:definition}
:label: hypothesisFunctionMulti
The __hypothesis__ or __mapping function__ for linear regression with $n$ features is:
\begin{equation*}
h_\theta(\boldsymbol{x^{(i)}}) = \theta_0 + \theta_1 x^{(i)}_1 + \theta_2 x^{(i)}_2 + \cdots + \theta_j x^{(i)}_j + \cdots + \theta_n x^{(i)}_n
\end{equation*}
````
```{warning}
There are $n+1$ parameters to optimize as we need to add the offset parameter $\theta_0$.
```
If we set $x_0^{(i)} = 1$, we can write the mapping function as a sum. For one training example $x^{(i)}$, i.e. a row in the data set:
```{math}
:label: h_theta_lin_sum
  h_\theta(\boldsymbol{x^{(i)}}) = \sum_{j=0}^n \theta_j x^{(i)}_j =  x^{(i)} \theta^{\; T}
```
where $\boldsymbol{x}^{(i)}$ a __row vector__ of $n+1$ elements, $x^{(i)} = (x^{(i)}_0, x^{(i)}_1, x^{(i)}_2, \cdots, x^{(i)}_n)$ and $\theta$ is a __row vector__ too of $n+1$ elements as well, $\boldsymbol{\theta} = (\theta_0, \theta_1, \cdots, \theta_n)$. Thus taking the transposed and multiplying by $\boldsymbol{x}^{(i)}$ is equivalent to a dot product:
\begin{equation*}
\boldsymbol{x^{(i)}} \boldsymbol{\theta^{\; T}} =  \begin{pmatrix} x^{(i)}_0, x^{(i)}_1, x^{(i)}_2, \cdots, x^{(i)}_n\\\end{pmatrix} 
  \begin{pmatrix} \theta_0 \\\theta_1 \\ \\\vdots \\  \\\theta_n \\\end{pmatrix} = \; \boldsymbol{x^{(i)}} \cdot  \boldsymbol{\theta}
\end{equation*}

(linRegMulti:gradDesc)=
## Gradient Descent in Multilinear Regression
We will revisit our algorithm to generalize it to $\theta_n$ parameters:

````{prf:algorithm} Gradient Descent for Multivariate Linear Regression
:label: gd_algo_multi
&nbsp;  
__Inputs__  
* Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $y$:
\begin{equation*}
X = \begin{pmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots  & x_j^{(1)} & \cdots & x_n^{(1)} \\[2ex]
x_1^{(2)} & x_2^{(2)} & \cdots & x_j^{(2)} & \cdots & x_n^{(2)} \\
\vdots  & \vdots & \ddots  & \vdots &  & \vdots \\
x_1^{(i)} & x_2^{(i)} & \cdots & x_j^{(i)} & \cdots & x_n^{(i)} \\
\vdots & \vdots &  & \vdots & \ddots  & \vdots \\
x_1^{(m)} & x_2^{(m)} & \cdots & x_j^{(m)} & \cdots & x_n^{(m)} \\
\end{pmatrix}  \hspace{10ex}  y = \begin{pmatrix}
y^{(1)} \\[2ex]
y^{(2)} \\[2ex]
\vdots  \\
y^{(i)}\\
\vdots \\[2ex]
y^{(m)}\end{pmatrix}
\end{equation*}
&nbsp;  
__Hyperparameters__  
* Learning rate $\alpha$
* Number of epochs $N$
&nbsp;  

__Outputs__  
The optimized values of model parameters $\boldsymbol{\theta} = \theta_0, \theta_1, \ldots , \theta_n$ minimizing $C(\boldsymbol{\theta})$.

1. __Initialization__: Set random values for all model parameters $\boldsymbol{\theta}$

1. __While the exit conditions are not met__:
   1. __Compute all partial derivatives of the cost function__:  
    &nbsp;  
    ```{math}
    :label: updateMulti
    \begin{equation*} \\
      \frac{\partial }{\partial \theta_j} C(\boldsymbol{\theta}) \;   \;   \;  \;   \;   \;  \forall j \in [0..n]
    \end{equation*}
    ```
   &nbsp;    
    &nbsp; &nbsp; &nbsp; 2. __Update all parameters__:  
    &nbsp; &nbsp; &nbsp; Use the update rule:  
    &nbsp;
    ```{math}
    :label: eqGDlinCostMulti
    \begin{equation*} \\
    \theta'_j = \theta_j-\alpha \frac{\partial}{\partial \theta_j} C\left(\boldsymbol{\theta}\right) \;   \;   \;  \;   \;   \; \forall j \in [0..n]\\
    \end{equation*}
    ```
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reassign the new $\boldsymbol{\theta}$ parameters to prepare for next iteration
```{math}
:label: eqUpdateThetaMulti
\begin{equation*}
\theta_j = \theta'_j \;   \;   \;  \;   \;   \; \forall j \in [0..n]\\
\end{equation*}
```

__Exit conditions__
* After the maximum number of epochs $N$ is reached
* If all derivatives tends to zero:
\begin{equation*}
\frac{\partial}{\partial \theta_j} C\left(\boldsymbol{\theta}\right) \rightarrow 0 \;   \;   \;  \;   \;   \;    \forall j \in [0..n]
\end{equation*}
````

With the linear hypothesis, the partial derivatives of $C(\boldsymbol{\theta})$ for each parameter $\theta_j$ will be of the form:
```{math}
:label: partialDevLinReg
\frac{\partial}{\partial \theta_j} C(\boldsymbol{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) -  y^{(i)}\right) \; x_j^{(i)}
```
(warmup:linregmulti:featurescalnorm)=
## Feature Scaling & Normalization
In the {ref}`previous sub-section on Graphical Visualization <linReg:GraphVisGD>`, the shown {ref}`3D plot of the cost function<plot_linReg_3D>` with respect to the $\boldsymbol{\theta}$ parameters was not really bowl-shaped but stretched. This happens when the features are of different ranges. One parameter will have an increment size larger than the other. 

```{admonition} Exercise
:class: seealso
If let's say $x_1$ has values between $10 000$ and $100000$ and $x_2$ between 1 and 10.  
1. Which parameter, $\theta_1$ or $\theta_2$ will be updated with a very large or very small step? (independently of the sign). Hint: ask yourself in which direction the contour is going to be stretched.
1. What are the consequences on the gradient descent algorithm?
```

````{admonition} Check your answers
:class: tip, dropdown
__Answer 1.__ If $x_1$ contains very large values, the coefficients $\theta_1$ will have to be very small and tuned in a fine way to not let the cost function explode. The $\theta_2$ parameters will not have to be tuned that finely to prevent the cost function from getting big. Thus the contour plot will be stretched along the parameter associated with the features spanning a small range. 

The $\theta$ parameters will descend (or converge) slowly on large ranges and quickly on small ranges, as the figure below shows:

```{figure} ../images/linReg_withandoutScaling.png
---
  name: withandoutScaling
  width: 100%
---
: The contour without feature scaling is skewed and takes up an oval shape. If normalized, the path towards the minimum is shorter.  
 <sub>Source: [enjoyalgorithms.com](https://www.enjoyalgorithms.com/blog/need-of-feature-scaling-in-machine-learning) </sub>
```

__Answer 2.__ The difference in range can lead to an 'overshooting' of one parameter value to the other side of the slope, thus creating a zig-zag path towards the minimum. It will slow down the learning process, because more steps are needed before converging to the minimum of the cost function. Another consequence is the risk of divergence: as the learning rate $\alpha$ is the same while updating all parameters, a big jump in one direction can even lead to instability (more on that in the next section).
````

### What is feature scaling?

````{prf:definition}
:label: defFeatureScaling
__Feature scaling__ is a data preparation process consisting of harmonizing the values of the input variables taken as features for a machine learning algorithm.
````

Feature scaling can be implemented in different ways. We will see the two classics: mean normalization and standardization.


### Mean normalization
````{prf:definition}
:label: defMeanNormalization
__Mean normalization__ consists of calculating the mean $\mu_j$ of all training examples $x_j^{(1)}, x_j^{(2)}, \cdots,  x_j^{(m)}$ of a given feature $j$ and subtract it to each example $x_j^{(i)}$ of that feature:
\begin{equation*}
\left(x^{(i)}_j\right)^\text{ norm} = x^{(i)}_j - \mu_j  \;   \;   \;  \;   \;   \;    \forall i \in [1..m]
\end{equation*}
It is usual to divide by the range of the features:
\begin{equation*}
\left(x^{(i)}_j\right)^\text{ norm} = \frac{x^{(i)}_j - \mu_j}{x_j^\max - x_j^\min} \;   \;   \;  \;   \;   \;    \forall i \in [1..m]
\end{equation*}
````

Consequence: the mean of the new normalized sample collection for that feature - think of it as an extra column in the matrix $X$ - will be zero. The data distribute the same as before, the values are just representing the _distance to the mean_.

### Standardization
````{prf:definition}
:label: defStandardization
__Standardization__ is a mean normalization procedure using as denominator the standard deviation $\sigma_j$ of all the samples for a given feature. 
\begin{equation*}
\widehat{x}^{(i)}_j = \frac{x^{(i)}_j - \mu_j}{\sigma_j} \;   \;   \;  \;   \;   \;    \forall i \in [1..m]
\end{equation*}
````
Consequence: the mean and standard deviation of the new normalized collection of feature $j$ will be zero and one respectively.

__When is feature scaling relevant? When is it not?__  
It all depends on how the data look like. This is why it is important before starting the machinery of (fancy) learning techniques to inspect the data, plot several distributions if possible and dedicate time for the preparedness steps: data cleaning, scaling etc. This is essential for the success of the fitting algorithm.

