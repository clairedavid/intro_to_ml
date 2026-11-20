# Challenging $k$-Means

An algorithm that always converges can still go wrong. 

## $k$-Means++: a smarter initialization

We've seen that the centroids are randomly picked at the start, using existing data points. This can cause issues if, for instance, two or more centroids are 'trapped' inside the same cluster. The algorithm may not converge toward the optimum but instead get stuck in a local minimum.

A way to cope? We can run $k$-Means several times and keep the run with the lowest cost $J$. 

Or we can run it once (more economical in resources) but be smarter with the initialization step by spreading the centroids out. This is the trick behind a modified version of the $k$-Means algorithm called $k$-Means++, introduced by Arthur and Vassilvitskii in 2007.

Here are the steps of $k$-Means++:

1. Choose an initial centroid $\boldsymbol{\mu}_1$ uniformly at random from $\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(m)} \in \mathbb{R}^{n}$ 
1. Compute the Euclidean distance of every remaining data point to the nearest centroid already chosen  
   \begin{equation}
   D\big(\boldsymbol{x}^{(i)}\big) = \min_{k} \, d\big(\boldsymbol{x}^{(i)}, \boldsymbol{\mu}_k\big)
   \end{equation}
1. Choose the next centroid $\boldsymbol{\mu}_k$, using weighted random sampling, where data points are drawn with a probability of 
   \begin{equation}
   P\big(\boldsymbol{x}^{(i)}\big) = \frac{D\big(\boldsymbol{x}^{(i)}\big)^2}{\sum_{l=1}^{m} D\big(\boldsymbol{x}^{(l)}\big)^2}
   \end{equation}
1. Repeat steps 2 and 3 until $K$ centroids are chosen
1. Proceed to the standard $k$-Means algorithm




It works by picking centroids sequentially, with a random draw where each point’s probability is its squared distance to its closest centroid, divided by the sum of the squared distances of all points. Why aren't we picking the furthest point as next centroid? If this point is an isolated outlier, it would be a wrong candidate for a centroid. Thanks to the weighted random sampling, all points far away from the already chosen centroids are likely to belong to another cluster and have more chances to be picked. Such a protocol at the start makes it unlikely that two centroids are picked inside the same cluster. With the standard initialization (random), centroids can still migrate during the iterations, but the risk of ending in a suboptimal local minimum is higher.


<div class="km-carousel" tabindex="0">
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_01.png" alt="k-Means++ example, frame 1 of 6"><p class="km-caption"><span class="km-count">1 / 6.</span> A small dataset of 12 unlabelled points. We can see 3 clusters, so we pick <i>K</i> = 3.</p></div>
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_02.png" alt="k-Means++ example, frame 2 of 6"><p class="km-caption"><span class="km-count">2 / 6.</span> The first centroid, <i>μ</i><sub>1</sub>, is drawn uniformly at random: every point has the same probability of being picked.</p></div>
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_03.png" alt="k-Means++ example, frame 3 of 6"><p class="km-caption"><span class="km-count">3 / 6.</span> For each remaining point, the squared distance to the nearest centroid, <i>D</i>(<b><i>x</i></b>)<sup>2</sup>, is computed. So far, there is only one: <i>μ</i><sub>1</sub>. The sum of all squared distances is 571.</p></div>
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_04.png" alt="k-Means++ example, frame 4 of 6"><p class="km-caption"><span class="km-count">4 / 6.</span> <i>μ</i><sub>2</sub> is drawn with probability <i>D</i>(<b><i>x</i></b>)<sup>2</sup>/571. Here the point (14, 11), with probability 101/571, is drawn as <i>μ</i><sub>2</sub>. It was the single most likely point. We see that points far away from the centroids are more likely to be drawn. That's the point, as they are more likely to belong to a different cluster.</p></div>
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_05.png" alt="k-Means++ example, frame 5 of 6"><p class="km-caption"><span class="km-count">5 / 6.</span> <i>D</i>(<b><i>x</i></b>)<sup>2</sup> is recomputed against the nearest of <i>μ</i><sub>1</sub> and <i>μ</i><sub>2</sub>. The top-right points are now close to <i>μ</i><sub>2</sub> and the sum of all squared distances is now 275 (it was 571 before). If we sum the squared distances of the points of the bottom cluster, we get 228. So there is a 228/275 ≈ 0.83 probability that the next centroid is picked among its points.</p></div>
  <div class="km-slide"><img src="../_static/kmeanspp/kmeanspp_06.png" alt="k-Means++ example, frame 6 of 6"><p class="km-caption"><span class="km-count">6 / 6.</span> <i>μ</i><sub>3</sub> is drawn: here, the point (7, 2) is picked, with probability 73/275 ≈ 0.27. All <i>K</i> starting centroids are chosen. The <i>k</i>-Means++ initialization is done. We can now start the standard <i>k</i>-Means algorithm.</p></div>
  <button class="km-btn km-prev" aria-label="Previous">&#9664;</button>
  <button class="km-btn km-next" aria-label="Next">&#9654;</button>
  <div class="km-dots"></div>
</div>

```{div} text-center
An illustrated example of $k$-Means++ with three clusters ($K = 3$).
```

This initialization works better than the standard (random) initialization. Actually, the authors managed to put an upper bound on the cost, on average, with respect to the best possible cost $J_\text{opt}$. The proof is found in the paper cited below. In short, they showed that the expectation of the cost of $k$-Means++ $J_{k\text{-Means++}}$ satisfies:
\begin{equation}
\mathbb{E}\big[J_{k\text{-Means++}}\big] \le 8(\ln K + 2)\, J_{\text{opt}}
\end{equation}

The $k$-Means++ algorithm is an improvement, but it is not perfect. There are caveats:

````{margin}
In `scikit-learn`, __mini-batch $\boldsymbol{k}$-Means__ is implemented via the `MiniBatchKMeans` class in the `sklearn.cluster` module. It speeds up clustering on massive datasets by updating cluster centroids using small, random subsamples (mini-batches) instead of the entire dataset on every iteration.

````
- __Sensitivity to outliers__: they may be picked as the initial centroids and alter the outcome. It is advised, if relevant of course, to clean the data beforehand.

- __Not scalable__: the initialization is sequential, with $K$ passes over the dataset. It can't be parallelized. This is a bottleneck slowing down the algorithm, especially for very large datasets. A fix? **$\boldsymbol{k}$-Means||** (pronounced as “$k$-means parallel”), where candidate centroids are sampled in rounds, executed in parallel. 


```{admonition} Learn More
:class: seealso
- __The $k$-Means++ paper__: [PDF from Stanford.edu](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)  
D. Arthur and S. Vassilvitskii, _$k$-means++: The Advantages of Careful Seeding_, Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2007.
- __Scalable K-Means++__: [arXiv:1203.6402](https://arxiv.org/abs/1203.6402)  
B. Bahmani, B. Moseley, A. Vattani, R. Kumar and S. Vassilvitskii, _Scalable K-Means++_, Proceedings of the VLDB Endowment, 5(7), 2012
```

## Knowing the number of clusters beforehand

With $k$-Means, we need to provide the number of clusters $K$ as input. But how do we know the number of clusters in advance? Most of the time, we do not. There are two methods helping us figure out the 'best $K$.'

First, we can run $k$-Means with several values of $K$ and use the cost as a proxy for evaluating the goodness of the clustering. The more clusters, the lower the cost will be. But there is a sharp descending slope at the start, until the curve flattens out. At that 'elbow,' we can find the number of clusters closest to the true $K$ (the notion of 'true $K$' is slightly misleading: there can be population clusters responding differently to a given drug and new clusters if more human groups are surveyed or more features added to the dataset).

Second, we can use the average silhouette coefficient, computed with all pairwise distances between all points in the dataset. This metric combines two measures: the cohesion (or compactness) and the separation. As you may guess, the former quantifies how close points within a cluster are. The latter assesses how far the point is from the nearest other cluster.

````{prf:definition}
:label: silhouettedef

The __silhouette coefficient__ of a point $i$ is defined as

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$ (silhouettecoefeq)

where

- $a_i$ is the __cohesion__: the average distance from point $i$ to the other points of its own cluster.
- $b_i$ is the __separation__: the average distance to the points of the nearest other cluster.

```` 

We want the cohesion to be small, ideally near 0. We want the separation to be large. If a cluster has only one point, the silhouette coefficient of that point is 0 by convention. From Equation {eq}`silhouettecoefeq`, the silhouette coefficient will range from $-1$ to $1$. A negative value is likely because the point is at the edge of its cluster and close to another one. We want the silhouette to be positive and as close to $1$ as possible.  

The average silhouette coefficient can be calculated within a cluster, by averaging the silhouette coefficients of points belonging to the cluster. It can also be computed over all points in the dataset, and this is where it becomes very handy to get a good value for $K$.

Let's consider the following dataset:

```{glue:figure} plot_kmeans_choosek_data
:name: choosekunlabelledscatter
:figwidth: 75%

Unlabelled dataset of 500 points. We have a rough idea of the true number of clusters.
```

{numref}`Figure %s <choosekcostsilhouette>` shows the cost (left) and the average silhouette coefficient over all points (right) for different values of $K$ (running $k$-Means each time).

```{glue:figure} plot_kmeans_choosek_curves
:name: choosekcostsilhouette

Resulting cost $J$ (left) and average silhouette coefficient (right) versus the number of clusters $K$ for the $k$-Means algorithm on the dataset shown in {numref}`Figure %s <choosekunlabelledscatter>`. 
```

The cost sharply drops and starts stabilizing around $K = 4$ and $K = 5$. For the silhouette, we get a peak for $K = 4$. Is this the right answer? Actually, the points were generated with 5 clusters, the bottom two overlapping. Here the silhouette could not pick this up. The cost curve does a better job, as a careful look at its variation could hint that 5 might be the last point before it flattens out. It's important to keep in mind that these plots only give insight into what the best number of clusters might be, not necessarily the truth.


```{admonition} Learn More
:class: seealso
- Chapter 7 on “Cluster Analysis: Basic Concepts and Algorithms”  
P.-N. Tan, M. Steinbach, A. Karpatne and V. Kumar, _Introduction to Data Mining_, 2019  
[Free chapter from the authors](https://www-users.cse.umn.edu/~kumar001/dmbook/ch7_clustering.pdf)   
```


## When $k$-Means fails

There are configurations where, no matter how clever the initialization and even with the right number of clusters, $k$-Means fails.

The algorithm struggles with clusters of different sizes or different densities, as shown in {numref}`Figures %s <kmeansfailsizes>` and {numref}`%s <kmeansfaildensities>`.

```{glue:figure} plot_kmeans_fail_sizes
:name: kmeansfailsizes
:figwidth: 90%

One large and two small clusters (left). The $k$-Means algorithm ($K = 3$) converges, but fails to distinguish the correct clusters: it cuts slices off the large one (right).
```

```{glue:figure} plot_kmeans_fail_densities
:name: kmeansfaildensities
:figwidth: 90%

One large sparse cluster and two small dense ones (left). The difference in density tricks the $k$-Means algorithm ($K = 3$): it converges, but merges the two dense clusters and splits the sparse one (right).
```

In summary, the $k$-Means algorithm is simple and powerful, yet not error-free. In the next section, we will introduce some alternatives.