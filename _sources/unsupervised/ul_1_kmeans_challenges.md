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
- __Sensitivity to outliers__: they may be picked as the initial centroids and alter the outcome.

- __Not scalable__: the initialization is sequential, with $K$ passes over the dataset. It can't be parallelized. This is a bottleneck slowing down the algorithm, especially for very large datasets. A fix? **$\boldsymbol{k}$-Means||** (pronounced “$k$-means parallel”), where candidate centroids are sampled in rounds, executed in parallel. 


```{admonition} Learn More
:class: seealso
- __The $k$-Means++ paper__: [PDF from Stanford.edu](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)  
D. Arthur and S. Vassilvitskii, _$k$-means++: The Advantages of Careful Seeding_, Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), 2007.
- __Scalable K-Means++__: [arXiv:1203.6402](https://arxiv.org/abs/1203.6402)  
B. Bahmani, B. Moseley, A. Vattani, R. Kumar and S. Vassilvitskii, _Scalable K-Means++_, Proceedings of the VLDB Endowment, 5(7), 2012
```

## Knowing the number of clusters beforehand



## When $k$-Means fails



