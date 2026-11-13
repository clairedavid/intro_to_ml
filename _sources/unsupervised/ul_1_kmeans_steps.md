# $k$-Means Step by Step

The simplest clustering algorithm. Featuring: Euclidean distance.


## Key concepts and one rule

Before diving into the procedure, let's define a central notion (pun intended).

````{prf:definition}
:label: centroiddef

A __centroid__ is a position in feature space representing the centre of a cluster. 

In the $k$-Means algorithm, the centroid corresponds to the centre of mass of the points assigned to it, i.e. its coordinates are the arithmetic means of each feature.

```` 

Other clustering algorithms use other definitions for 'centre':
- $k$-medoids uses an actual data point minimising the total distance to all other points in the cluster
- $k$-medians uses the median of each feature, relying typically on the Manhattan distance (L1 norm) as similarity measure.


Now that we know what a centroid is, we can state the one and only rule behind the clustering algorithm:

```{admonition} $k$-Means Rule
:class: tip

Each point is assigned to the cluster whose centroid is the closest.
```


## An illustrated example

Let's go through a visual and simple case to intuitively present how $k$-Means works.

(kmeanstour)=
<div class="km-carousel" tabindex="0">
  <div class="km-slide"><img src="../_static/kmeans/kmeans_01_points.png" alt="k-Means example, frame 1 of 8"><p class="km-caption"><span class="km-count">1 / 8.</span> Initialization. We need to know the number of clusters beforehand: How many clusters here?</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_02_dataset.png" alt="k-Means example, frame 2 of 8"><p class="km-caption"><span class="km-count">2 / 8.</span> We roughly see two clusters. We can pick two random data points and set them as 'centroids' to start the algorithm. Here let's pick A (4, 10) and B (6, 11).</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_03_init.png" alt="k-Means example, frame 3 of 8"><p class="km-caption"><span class="km-count">3 / 8.</span> Now let's assign all the data points to either <i>μ</i><sub>1</sub> (red cluster) or <i>μ</i><sub>2</sub> (green cluster) following the <i>k</i>-Means rule.</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_04_iter1_assign.png" alt="k-Means example, frame 4 of 8"><p class="km-caption"><span class="km-count">4 / 8.</span> Points are coloured (assigned): the so-called E-step (expectation) is done. Is it over? No! We need to update our centroids to put them at the centre of mass of their cluster.</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_05_iter1_update.png" alt="k-Means example, frame 5 of 8"><p class="km-caption"><span class="km-count">5 / 8.</span> M-step: let's recompute the centroid positions to minimize the sum of squared distances of each cluster point with respect to their centroid (more on why it is called M-step later). Are we done? No! The rule is not followed: point E is closer to <i>μ</i><sub>2</sub> than to <i>μ</i><sub>1</sub>, and point B is closer to <i>μ</i><sub>1</sub> than to <i>μ</i><sub>2</sub>. They need to switch clusters.</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_06_iter2_assign.png" alt="k-Means example, frame 6 of 8"><p class="km-caption"><span class="km-count">6 / 8.</span> E-step again: we reassign each point to the cluster matching its closest centroid. Point E is now green and B red. Good. Are we done? No! We need to recompute the centroids' positions again.</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_07_iter2_update.png" alt="k-Means example, frame 7 of 8"><p class="km-caption"><span class="km-count">7 / 8.</span> M-step again: the centroids are moved to the centre of mass of their cluster.</p></div>
  <div class="km-slide"><img src="../_static/kmeans/kmeans_08_converged.png" alt="k-Means example, frame 8 of 8"><p class="km-caption"><span class="km-count">8 / 8.</span> Are we done? Yes! There is no reassignment of points to do. Nothing has changed. The algorithm has converged.</p></div>
  <button class="km-btn km-prev" aria-label="Previous">&#9664;</button>
  <button class="km-btn km-next" aria-label="Next">&#9654;</button>
  <div class="km-dots"></div>
</div>

```{div} text-center
A step-by-step illustrated example of the $k$-Means algorithm with two clusters ($K = 2$).
```

The algorithm is simple: it alternately updates the assignment of data points to clusters and the centroid positions. We can't go wrong (well, we will soon see that it can lead to bad results). As each step can only improve -- or keep as is -- the clustering and as there is a finite number of possibilities to assign $m$ data points to $k$ clusters, the algorithm always converges.

Let's write it down in a more mathematical way.

## $k$-Means: a (very) simple case of EM algorithm

Here we have two unknowns: the assignment of each data point to a given cluster and the positions of the centroids of the clusters. We saw from the illustrated example that we cannot optimize for both at the same time. We need to fix one, optimize the other and vice versa. And repeat until convergence. 

The $k$-Means algorithm can be seen as a very simple case of an EM algorithm, a big family of iterative methods in probabilistic machine learning. The E stands for Expectation, and M for Maximization. We won't go into details but the main idea is the E-step gives a function for the expectation of a log-likelihood. The M-step is about maximizing the log-likelihood by tweaking its parameters. Then those optimized parameters are defining a new expectation function used for the next E-step.

Here it is much simple. There is no probability distribution: our centroid positions are deterministic (or if you really want to see a Gaussian, think of one with a standard deviation of zero and mean centred on a given centroid's position). 

The $k$-Means algorithm alternates bewteen:

* __Expectation-step__:  
  Centroid positions fixed; assign each point to its nearest centroid
* __Maximization-step__:  
  Fix the assignment points-cluster; move each centroid to the mean of its cluster.

But wait: what is being maximized here?  

This brings us to the definition of the cost. Yes, again, there is a cost in the story.


## The associated cost

In the previous chapters on supervised learning, we saw algorithms with a cost to minimize, let it be the MSE, or one using the Gini index. Here we only find the closest centroid for each data point, and compute the mean of the data points belonging to a given cluster. Where is the cost? It is actually hidden.


````{prf:definition}
:label: kmeanscostdef

In $k$-Means, each iteration reduces the __within-cluster sum of squared distances__: 

$$
J(c, \boldsymbol{\mu}) = \sum_{i=1}^{m} \left\lVert \boldsymbol{x}^{(i)} - \boldsymbol{\mu}_{c^{(i)}} \right\rVert_2^2
= \sum_{k=1}^{K} \sum_{i \in \mathcal{C}_k} \left\lVert \boldsymbol{x}^{(i)} - \boldsymbol{\mu}_k \right\rVert_2^2
$$ (kmeanscosteq)

where

- $\boldsymbol{\mu} = (\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K)$ are the $K$ centroids, with $\boldsymbol{\mu}_k \in \mathbb{R}^n$,
- $c = (c^{(1)}, \dots, c^{(m)})$ are the cluster assignments, with $c^{(i)} \in \{1, \dots, K\}$ the cluster of data point $\boldsymbol{x}^{(i)}$,
- $\mathcal{C}_k = \{\, i : c^{(i)} = k \,\}$ is the set of indices of the data points assigned to cluster $k$.

````

$k$-Means is a minimization problem: its main objective is to minimize the sum of the squared distances between the points and their respective cluster centroid. 
The cost can been seen as one sum, over all data points, computing the square distance between each data point and its associated cluster. Or equivalently, a sum of sums, per cluster, where the squared distances of all points from a given cluster are first added. And then adding all these $K$ sums together.

Why aren't we using the cost in the algorithm? We do in a way: computing the mean is exactly the point that minimizes $J(c, \boldsymbol{\mu})$ for a given cluster. 

The cost is useless then? Not really, it serves for evaluation. Especially if the algorithm falls into a local minimum due to e.g. an unfortunate centroid initialization (more on that later).

````{grid} 2
:gutter: 3

```{grid-item-card} E-step
:text-align: center

$$
c^{(i)} = \underset{k \in \{1, \dots, K\}}{\arg\min} \left\lVert \boldsymbol{x}^{(i)} - \boldsymbol{\mu}_k \right\rVert_2^2
$$

Centroids $\boldsymbol{\mu}_k$ given, filling cluster assignment $c^{(i)}$.
```

```{grid-item-card} M-step
:text-align: center

$$
\frac{\partial J}{\partial \boldsymbol{\mu}_k} = 0
\;\Longrightarrow\;
\boldsymbol{\mu}_k = \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} \boldsymbol{x}^{(i)}
$$

The mean is exactly the point that minimizes $J$ for a given cluster.
```

````


## The pseudo-code

This is left as an exercise, and an opportunity to practice pseudocode writing!


````{admonition} Exercise
:class: seealso
From the illustrated step-by-step example and the equations above, write the $k$-Means pseudocode.
````

````{admonition} To get started
:class: hint, dropdown
Write first the general skeleton:
* What are the inputs?
* What are the outputs?
* No spoiler: we know it is iterative, so where and when/how does the loop start and stop?
````

````{admonition} Another hint: a skeleton
:class: hint, dropdown
Here is the general structure of the pseudocode for $k$-Means without math.

```{prf:algorithm} k-Means skeleton
:nonumber:

$$
\begin{array}{rl}
\text{1:} & \text{Select } K \text{ data points as initial centroids.} \\
\text{2:} & \textbf{repeat} \\
\text{3:} & \quad \text{Form } K \text{ clusters by assigning each point to its closest centroid.} \\
\text{4:} & \quad \text{Recompute the centroid of each cluster.} \\
\text{5:} & \textbf{until } \text{the centroids do not change.}
\end{array}
$$
```

Adapted from Tan et al., *Introduction to Data Mining* (2nd ed.), Algorithm 7.1.

Now you have to insert the math!
````

````{admonition} Check your answer
:class: tip, dropdown

```{prf:algorithm} $k$-Means
:label: kmeansalgo

$$
\begin{array}{rl}
 & \textbf{Input: } \text{data } \boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(m)} \in \mathbb{R}^{n} \\
 & \textbf{Hyperparameters: } \text{number of clusters } K, \text{ maximum iterations } N_{\max} \\
 & \textbf{Output: } \text{assignments } c^{(1)}, \dots, c^{(m)}, \text{ centroids } \boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K \\[6pt]
\text{1:} & \text{Choose } K \text{ distinct data points at random as } \boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K \\
\text{2:} & \textbf{repeat} \\
\text{3:} & \quad \textbf{for } i = 1, \dots, m \textbf{ do} \\
\text{4:} & \qquad c^{(i)} \leftarrow \underset{k \in \{1, \dots, K\}}{\arg\min} \left\lVert \boldsymbol{x}^{(i)} - \boldsymbol{\mu}_k \right\rVert_2^2 \\[4pt]
\text{5:} & \quad \textbf{for } k = 1, \dots, K \textbf{ do} \\
\text{6:} & \qquad \mathcal{C}_k \leftarrow \{\, i : c^{(i)} = k \,\} \\
\text{7:} & \qquad \boldsymbol{\mu}_k \leftarrow \dfrac{1}{|\mathcal{C}_k|} \displaystyle\sum_{i \in \mathcal{C}_k} \boldsymbol{x}^{(i)} \\[4pt]
\text{8:} & \textbf{until } \text{no } c^{(i)} \text{ changes } \textbf{or } N_{\max} \text{ iterations reached} \\
\text{9:} & \textbf{return } c^{(1)}, \dots, c^{(m)} \text{ and } \boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K
\end{array}
$$
```
````


It all looks beautiful. But $k$-Means has its limitations. Some we can address, others we cannot, as we are about to see in the next section.