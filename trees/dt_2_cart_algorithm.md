# The CART Algorithm
The algorithm used for classification trees is called CART. It stands for Classification and Regression Tree. 

## Gini the metric

There are metrics entering in the optimization algorithm of a decision tree. The main one is the Gini impurity:

````{prf:definition}
:label: ginidef
The __Gini's diversity index__ is a measure of a node's impurity.  

It is defined for each tree node $i$ as:

\begin{equation}
G_i = 1 - \sum_{k=1}^{N_\text{classes}} \left( \frac{N_{k, i}}{ N_i} \right)^2 = 1 - \sum_{k=1}^{N_\text{classes}} \left( p_k \right)^2
\end{equation}

with $N_{k, i}$ the number of data samples of class $k$ in node $i$ and $N_{i}$ the total number of data samples in node $i$. The terms $p_k$ in the sum are equivalent to the probability of getting a sample of class $k$ in the node $i$.

The Gini's impurity index ranges from 0 (100% pure node) to 1 (very impure node).
````

If a node has an equal amount of data points from two classes, let's say 500 points each, the Gini index will be:

\begin{equation}
G_i = 1 - \left( \frac{500}{1000} \right)^2 - \left( \frac{500}{1000} \right)^2 = 0.5 
\end{equation}

With three classes present in equal quantities, it will be $\frac{2}{3}$, for four, $\frac{3}{4}$, etc. 

## The cost function in CART

Once again, we will need a cost function to minimize. This cost function will be defined at a given node containing $N_\text{node}$ data samples.

````{prf:definition}
:label: costCART
The cost function for CART is defined as:

\begin{equation}
C(j, t_j) = \frac{N_\text{left}}{N_\text{node}} G_\text{left} + \frac{N_\text{right}}{N_\text{node}} G_\text{right}
\end{equation}

````
It is a sum where the purity of each subset is weighted by the subset sizes. In the literature, you may see the terms left and right. They translate as:

- left: all data points whose feature $x_j$ is such that $x_j < t_j$ 
- right: all data points whose feature $x_j$ is such that $x_j \geq t_j$ 

We will see the role of the hyperparameters later. For now, let's now go through the steps of the algorithm.

## How CART works

````{prf:algorithm} Classification and Regression Tree
:label: DTalgo

__Inputs__  
* Training data set $X$ of $m$ samples with each $n$ input features, associated with their targets $y$
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

__Hyperparameters__  
* Max Depth 
* Minimum sample per split
* Minimum samples per leaf
* Maximum number of leaf nodes  
(non-exhaustive list)


__Outputs__  
A collection on decision boundaries partitioning the feature space.


__Initialization:__ at the root node


__STEP 1: THRESHOLD COMPUTATION__:  

__For__ each feature $j = 1, \ldots, n$ :  
$\;\;\;\;\;\;\;$ __For__ each candidate threshold $t_j$ scanning the range $x^\text{min}_j$ to $x^\text{max}_j$:  
$\;\;\;\;\;\;\;\;\;\;\;\;\;\;$ Compute the cost function $C(j, t_j)$  
&nbsp;  
$\;\;\;\;\;\;\;$ Find for this feature the threshold $t_j^*$ associated with the minimum cost:

\begin{equation}
t_j^* = \arg\min_{t_j} C(j, t_j)
\end{equation}

After scanning all features $j$, find the tuple $(j^{\text{cut}}, t_{j^{\text{cut}}}^*)$ that minimizes:  
\begin{equation}
(j^{\text{cut}}, t_{j^{\text{cut}}}^*) = \arg\min_{j,\, t_j^*} C(j, t_j^*)
\end{equation}

__STEP 2: BRANCHING__:  
Split the dataset along the feature $j^{\text{cut}}$ using the threshold $t_{j^{\text{cut}}}^*$ into two subsets  
(left and right child nodes)

Repeat Step 1 at each new node.

__Exit conditions__  
Stop when at least one of the hyperparameter constraints is satisfied.

````

One of the greatest strength of decision tree is the fact very few assumptions about the training data are made. It differs from linear or polynomial models where we need to know beforehand that the data can be fit with either linear or polynomial function.  

Another advantage: feature scaling is not necessary. No need to standardize the dataset beforehand.

Decision trees are also easy to interpret. It's possible to check calculations and apply the classification rules even manually. Such clarity in the algorithm, often called _white box_ models, contrasts with the _black boxes_ that are for instance neural networks. 

Despite these advantages, decision trees have several limitations.

