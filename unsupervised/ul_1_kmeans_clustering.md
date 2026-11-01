# What is clustering?

## Definition

````{prf:definition}
:label: clusteringdef

__Clustering__ is the task of grouping unlabelled data into subsets called __clusters__, according to a similarity measure.

__Clusters__ are groups of data points sharing a similarity, i.e. points in the same cluster are more similar to each other than to points in other clusters.

````

```{glue:figure} plot_clustering_intro
:name: clustering_intro
:figwidth: 100%

Unlabelled data (left) are grouped into clusters (right).
```

In a sense, clustering labels data, although the meaning of each cluster is not known in advance.

Clustering is often a first step to explore a dataset. It is also referred to as "cluster analysis" in the literature (usually in statistics and social sciences). Note that clustering does not necessarily imply the use of machine learning techniques, although the term is now mostly employed in the context of ML. 




## Similarity measures

What does it mean for two data points to be similar?


### The Euclidean distance

The most obvious metric of similarity is closeness, referring to the "shortest distance" between data points in the feature space. The mathematical translation of a straight-line distance between two data points is the Euclidean distance.

````{margin}
```{warning} 
Because differences are squared, variables with high values will dominate the distance calculation. It is essential here to standardize or normalize the data before proceeding.
```
````

````{prf:definition}
:label: euclideandistancedef

The __Euclidean distance__ measures the length of the shortest path between two points in an $n$-dimensional space.

Given two vectors $\mathbf{u} = (u_1, \dots, u_n)$ and $\mathbf{v} = (v_1, \dots, v_n)$ in $\mathbb{R}^n$, it is defined as

$$
d(\mathbf{u}, \mathbf{v}) = \lVert \mathbf{u} - \mathbf{v} \rVert_2 = \sqrt{\sum_{j=1}^{n} \left(u_j - v_j\right)^2}
$$

where $u_j - v_j$ is the difference between the two vectors along the $j$-th feature.

````
You can recognise the 2-norm in {prf:ref}euclideandistancedef; in data science references, the Euclidean distance is often called the L2 distance.

We will see in the following section that this distance is at the core of the $k$-Means algorithm (you may already guess why from its name).



### The cosine similarity

Multidimensional data can be seen as a collection of data points, but also of vectors departing from a central point and with as many components as features. What if some vectors from a dataset have very different norms but are aligned in the same direction? They would not be similar in scale but definitely close in their angular separation. The idea behind cosine similarity is to measure that directional closeness. It is widely used in data mining and recommender systems.

