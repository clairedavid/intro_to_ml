# Multiclass Classification 
A multi-class classification problem can be split into multiple binary classification datasets and be trained as a binary classification model each. Such approach is a heuristic method, that is to say not optimal nor direct. But it eventually does the job.

Let's for instance consider three classes, labelled with their colours and distributed in two dimensions (two input features) like this:

```{figure} ../images/logReg_multiclass-1.webp
---
  name: logReg_multiclass-1
  width: 60%
---
: 2D distribution of three different classes.  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```


There are two main approaches of such methods for multiclass classification.

## One-to-One approach

````{prf:definition}
:label: multiclass1to1def

The __One-to-One approach__ consists of applying a binary classification for each pair of classes, ignoring the other classes.

With a dataset made of $N^\text{class}$ classes, the number of models to train, $N^\text{model}$ is given by 
\begin{equation}
N^\text{model} = \frac{N^\text{class}(N^\text{class}-1)}{2}
\end{equation}

Each model predicts one class label. The final decision is the class label receiving the most votes, i.e. being predicted most of the time.
````


The One-to-One method would create those hyperplanes (with two input features, D = 2 we will have a 1D line as separation):

```{figure} ../images/logReg_multiclass-2.webp
---
  name: logReg_multiclass-2
  width: 60%
---
: One-to-One approach splits paired datasets, ignoring the points of the other classes.  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```

__Pro__  
The sample size is more balanced between the two chosen classes than if datasets were split with one class against all others.

__Con__  
The pairing makes the number of models to train large and thus computer intensive.


## One-to-All approach

````{prf:definition}
:label: multiclass1toalldef

The __One-to-All or One-to-Rest approach__ consists of training each class against the collection of all other classes.

With a dataset made of $N^\text{class}$ classes, the number of pairs to train is
\begin{equation}
 N^\text{model} = N^\text{class}
\end{equation}


The final prediction is given by the highest value of the hypothesis function $h^{k}_\theta(x)$, $k \in [1, N^\text{model}]$ among the $N^\text{model}$ binary classifiers.

````


```{figure} ../images/logReg_multiclass-3.webp
---
  name: logReg_multiclass-3
  width: 60%
---
: One-to-All approach focuses on one class to discriminate from all other points  
 (i.e. all other classes are merged into a single 'background' class).  
 <sub>Image: [analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)</sub>
```

__Pro__  
Less binary classifiers to train.

__Con__  
The number of data points from the positive/signal class will be very small if the negative/background class is the merging of all other data points from the other classes. The model may fail to learn the patterns that identify the rare positive class because it is penalized so little for misclassifying it.


## Further reading

Some further reading if you are curious:

```{admonition} Learn more
:class: seealso
* [One-vs-Rest and One-vs-One for Multi-Class Classification, machinelearningmastery.com](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)
* [Multiclass Classification Using SVM, analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/05/multiclass-classification-using-svm/)
```

