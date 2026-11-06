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
You can recognise the $\ell_2$ norm in {prf:ref}`euclideandistancedef`; in data science references, the Euclidean distance is often called the L2 distance.

We will see in the following section that this distance is at the core of the $k$-Means algorithm (you may already guess why from its name).



### The cosine similarity

Multidimensional data can be seen as a collection of data points, but also of vectors departing from the origin and with as many components as features. What if some vectors from a dataset have very different norms but are aligned in the same direction? They would not be similar in scale but definitely close in their angular separation. The idea behind cosine similarity is to measure that directional closeness. It is widely used in text mining and recommender systems.

Let's take an example in text mining. If the dataset lists the word count of many words for different books and articles, a thousand-page book about particle physics -- random topic 😉 -- will have a large number of occurrences of the word _particle_. A short article also about particle physics will have far fewer occurrences. Yet those two documents are on the same topic. In that sense, the magnitude, in our example the raw number of occurrences, does not matter. What is important is the ratio of certain words, like _particle_, over others, such as _elephant_, which would be frequent in a book about biology.

The cosine similarity helps define "angular clusters" in the dataset, as seen below.  


```{glue:figure} plot_cosine_clusters
:name: plot_cosine_clusters
:figwidth: 100%

Unlabelled dataset (left) clustered according to cosine similarity (right).
```

Note that in our text mining example above, the features would be non-negative, hence we would only have data points in the first quadrant. 


````{prf:definition}
:label: cosinesimidef

The __cosine similarity__ is a measure of how similar the directions of two non-zero vectors $\mathbf{u}$ and $\mathbf{v}$ in $\mathbb{R}^n$ are.

It is the cosine of the angle $\theta$ between the two vectors:

$$
S_C(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert_2 \, \lVert \mathbf{v} \rVert_2}
= \frac{\sum_{j=1}^{n} u_j \, v_j}{\sqrt{\sum_{j=1}^{n} u_j^2} \; \sqrt{\sum_{j=1}^{n} v_j^2}}
$$ (cosinesimilarityeq)

where $\mathbf{u} \cdot \mathbf{v}$ is the dot product of the two vectors and $\lVert \cdot \rVert_2$ their $\ell_2$ norm.

````

As the definition states, the cosine similarity cannot be calculated with a zero vector. It is thus important to check the dataset beforehand to make sure all vectors are non-zero.

What values can it take? The cosine lies between $-1$ and $1$, so:

- $S_C = 1 \;\Leftrightarrow\; \theta = 0^\circ$: the vectors point in exactly the same direction. They are highly similar.
- $S_C = 0 \;\Leftrightarrow\; \theta = 90^\circ$: the vectors are perpendicular, or orthogonal. This indicates no relationship or shared context.
- $S_C = -1 \;\Leftrightarrow\; \theta = 180^\circ$: the vectors point in diametrically opposite directions. They are completely dissimilar.

If features are non-negative, such as a word count, then the range is from 0 to 1.

```{admonition} Learn More
:class: seealso

* Curious about other distance measures? The SciPy documentation is here for you: [Distance computations (`scipy.spatial.distance`)](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
* [4 Distance Measures for Machine Learning](https://machinelearningmastery.com/distance-measures-for-machine-learning/) (machinelearningmastery.com): definitions and code snippets

```