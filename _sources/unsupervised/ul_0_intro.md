# Learning Without Labels?

<br>

<br>

Unsupervised learning is a vast family of techniques identifying relationships in unlabelled data. Here, the dataset is purely made of the input features. No associated targets. No record of the "truth." The immediate question we can ask ourselves is then:

```{div} text-center
How do we learn?
```

As much as I stress that questions are a good guide to progress in one's understanding, this one is actually misleading. It implicitly assumes that labels are necessary. They are not. Patterns can be found in the data without targets. The question we now ought to ask is rather:

```{div} text-center
What can we learn?
```

A dataset, in particular a large one, contains information on its own. The goal of unsupervised learning is to discover hidden patterns within the data. They are "hidden" because datasets are unlabelled and most often high-dimensional, so they are not directly obvious. We will see the types of patterns we can reveal. 

This chapter offers a very short introduction to unsupervised learning. This sub-domain is so rich that it would deserve an entire course on its own. We will go over two tasks: clustering and dimensionality reduction, presenting the most popular techniques as foundational knowledge. 

<br>

<br>
