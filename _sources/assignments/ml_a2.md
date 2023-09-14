# Assignment 2: Decision Trees

````{margin}
```{admonition} Dataset
Find the repository on GDrive [here](https://drive.google.com/drive/folders/1b_GDA2bfsUhlvzX-A7RjHoHCL5Z8-bkb?usp=sharing).
```
````
## 1. Decision Stump by hand
Using the data from tutorial 2, you will implement a one-level decision tree, or decision stump.  
You will use the CART algorithm (Classification And Regression Tree). Recall the Gini's index measuring the impurity is defined as:  
\begin{equation*}
G_i = 1 - \sum_{k=1}^{N_\text{classes}} \left( \frac{N_{k, i}}{ N_i} \right)^2 
\end{equation*}
The cost function is
\begin{equation*}
J(k, t_k) = \frac{n_\text{left}}{n_\text{node}} G_\text{left} + \frac{n_\text{right}}{n_\text{node}} G_\text{right} \;,
\end{equation*}
where $k$ is a given feature and $t_k$ the threshold on that feature. We will use $|\Delta\eta_{jj}|$ and $m_{jj}$ as our two input variables. The main function `decision_stumper` should return the optimized threshold and cost function values for a given feature. It should take as arguments:
* the dataframe
* the variable name of the input feature 
* the class name (column name where labels are stored)
* the class values (in an array)
* the numbers of threshold values swiping the interval of the feature

__1.1 Get and load the data__  
Get the `train` dataset and load the relevant columns in a dataframe.

__1.2 Compute the Gini index__    
Write a function computing the Gini index value. Make your code as general as possible.  
Add in the next cell a series of tests.  
_Bonus: secure your code to prevent a division by zero._

__1.3 Calculate the cost__  
Write a function computing the cost function in the CART algorithm. 

__1.4 Main function: code a Decision Stump__  
Write the main function `decision_stumper` that will call the functions defined above. Call your function on each input feature and conclude on the final cut for your decision stump.
````{margin}
```{tip}
If you cannot complete the previous question, pick one of the two features and manually choose a threshold yourself to be able to draw a decision boundary.
```
````
You just coded a decision stump by hand!

__1.5 Plot the cut__  
Use the `plot_scatter` function from the second tutorial and modify it to draw the line corresponding to the optimized threshold from the decision stump. You can use Matplotlib's `axhline` or `axvline` method for drawing a horizontal or vertical line respectively. Try to be as general as possible in the input arguments.  
_Hints provided on demand during office hours._

## 2. Plotting mission: the overtraining check
The goal of this exercise is to understand and reproduce the following plot:

```{figure} ../images/a02_overtrainingTMVA2.png
---
  name: a02_overtrainingTMVA2
  width: 80%
---
The "overtraining check" plot from [TMVA](https://root.cern/manual/tmva/) (MVA stands for MultiVariate Analysis), a library used in High Energy Physics (HEP) within the [ROOT](https://root.cern/) framework.  
 <sub>Image: root-forum, cern.ch</sub>
```

The $x$-axis is an equivalent to the output scores of a classifier (here it is a Support Vector Machine, which are not covered in this course). 

We will use the same dataset as for the tutorial 2 'Forestree.' However as the validation set lacks statistics, you will use the training and __testing__ sets, the latter having more samples than the validation set.

__2.1 Understanding of the plot__  
Describe the plot and explain why this is called an "overtraining check" plot. Importance will be given to the clarity of your answer.  

````{margin}
```{important}
Your code should be clear and relevantly commented. 
```
````
__2.2 Reproducing the plot__  
Write a function `plot_overtraining_check` that takes as arguments the classifier object, the $X$ and $y$ lists of the training and testing sets, the value of the positive class (e.g. for VBF it is 1) and a title. The function should split each dataset (train/test) into the real category (signal/background).

To test your plotting macro, use a Decision Tree classifier of maximum depth 2. You will obtain a plot like this:

```{figure} ../images/a02_overtraining_plt_DTmaxD2.png
---
  name: a02_overtraining_plt_DTmaxD2
  width: 100%
---
Example of an overtraining check plot using Matplotlib.
```
```{warning}
The histograms on the overtraining check plot are normalized to an area of unity. In Matplotlib, this is done using the argument `density = 1`. However you should be very careful to rescale properly the vertical error bars before plotting the testing set.
```

__2.3 Using the plot__  
Create a random forest classifier with 100 estimators and leave other hyperparameters as default. Plot the overtraining check with this classifier. What are your observations? Is it classifying well on the training set? Is is under- or overtrain? Why? 

Create a second random forest classifier with this time the option `max_leaf_nodes=32`.  What is improved? What is still problematic? 

```{important}
&nbsp;  
You are encouraged to work in groups, however submissions are individual.

If you have received help from your peers and/or have worked within a group, summarize in the header of your notebook the involved students, how they helped you and the contributions of each group member. This is to ensure fairness in the evaluation.

You can use the internet such as the official pages of relevant libraries, forum websites to debug, etc. However, using an AI such as ChatGPT would be considered cheating (and not good for you anyway to develop your programming skills).

The instructor and tutors are available throughout the week to answer your questions. Write an email with your well-articulated question(s). Put in CC your teammates if any.

Thank you and do not forget to have fun while coding!
```