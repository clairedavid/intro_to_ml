# Assignment 3: Neural Network By Hand!

In this assignment, you will code a little neural network from scratch! Don't worry, it will be extremely guided. The two most important points being that you learn and you have fun.

We want our neural network to solve the XOR problem. This is how the data look like:
```{figure} ../images/a03_data.png
---
  name: a03_data
  width: 70%
---
Scatter plot of data representing the XOR configuration: it is not linearly separable.
```



## 0. Setup
Copy all of the following into a fresh Jupyter notebook.

```python
import numpy as np
import pandas as pd
import random
import math
np.random.seed(42)

import matplotlib.pyplot as plt
FONTSIZE = 16
params = {
         'figure.figsize' : (6,6),
         'axes.labelsize' : FONTSIZE,
         'axes.titlesize' : FONTSIZE+2,
         'legend.fontsize': FONTSIZE,
         'xtick.labelsize': FONTSIZE,
         'ytick.labelsize': FONTSIZE,
         'xtick.color'    : 'black',
         'ytick.color'    : 'black',
         'axes.facecolor' : 'white',
         'axes.edgecolor' : 'black',
         'axes.titlepad'  :  20,
         'axes.labelpad'  :  10}
plt.rcParams.update(params)

XNAME = 'x1'; XLABEL = r'$x_1$'
YNAME = 'x2'; YLABEL = r'$x_2$'
RANGE = (-6, 6); STEP = 0.1

def predict(output_node, boundary_value):
  output_node.reshape(-1, 1, 1) # a list (m, 1, 1)
  predictions = np.array(output_node > boundary_value, dtype=int)
  return predictions

def plot_cost_vs_iter(train_costs, test_costs, title="Cost evolution"):

  fig, ax = plt.subplots(figsize=(8, 6))
  iters = np.arange(1,len(train_costs)+1)
  ax.plot(iters, train_costs, color='red', lw=1, label='Training set')
  ax.plot(iters, test_costs, color='blue', lw=1, label='Testing set')
  ax.set_xlabel("Number of iterations"); ax.set_xlim(1, iters[-1])
  ax.set_ylabel("Cost")
  ax.legend(loc="upper right", frameon=False)
  ax.set_title(title)
  plt.show()


def get_decision_surface(weights, biases, boundary=0.5, range=RANGE, step=STEP):

  # Create a grid of points spanning the parameter space:
  x1v, x2v = np.meshgrid(np.arange(range[0], range[1]+step, step),
                         np.arange(range[0], range[1]+step, step))
  
  # Stack it so that it is shaped like X_train: (m,2)
  X_grid = np.c_[x1v.ravel(), x2v.ravel()].reshape(-1,2)

  # Feedforward on all grid points and get binary predictions:
  output = feedforward(X_grid, weights, biases)[-1] # getting only output node
  Ypred_grid = predict(output, boundary)

  return (x1v, x2v, Ypred_grid.reshape(x1v.shape))


def plot_scatter(sig, bkg, ds=None, xname=XNAME, xlabel=XLABEL, yname=YNAME, ylabel=YLABEL, range=RANGE, step=STEP, title="Scatter plot"):

  fig, ax = plt.subplots()

  # Decision surface
  if ds:
    (xx, yy, Z) = ds # unpack contour data
    cs = plt.contourf(xx, yy, Z, levels=[0,0.5,1], colors=['orange','dodgerblue'], alpha=0.3)

  # Scatter signal and background:
  ax.scatter(sig[xname], sig[yname], marker='o', s=10, c='dodgerblue', alpha=1, label='Positive class')
  ax.scatter(bkg[xname], bkg[yname], marker='o', s=10, c='orange',     alpha=1, label='Negative class')

  # Axes, legend and plot:
  ax.set_xlim(range); ax.set_xlabel(xlabel)
  ax.set_ylim(range); ax.set_ylabel(ylabel)

  ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", frameon=False) 
  ax.set_title(title)
  plt.show()
```

## 1. Get the data
__1.1: Get the data file__  
Retrieve the files of [this folder](https://drive.google.com/drive/folders/1qJaNCiZ6FTOltLuOoMU1sZWmNtgbxlcN?usp=sharing) and read them into dataframes `train` and `test`.
* How many samples are there in each? Write a full sentence to express your answer.
* What is the name of the column containing the labels? What are the class values?

__1.2: Split signal vs background__  
Create a `sig` and `bkg` dataframes that collect the real signal and value samples.  
_Hint: this has been done in Tutorial 2._

__1.3: Dataframe to NumPy__  
Declare the following variables to store the dataset into NumPy arrays:
```python
inputs = ['x1', 'x2']
X_train = 
y_train = 

X_test = 
y_test = 
```

## 2. Functions
__2.1: Weighted Sum__  
Create the function `z` that compute the weighted sum of a given activation unit, as seen in class. You do not have to worry about the dimensions of objects here, this will come later. Make sure the names of the arguments are self-explanatory:
```python
def z( ... , ... , ... ):
    # ...
    return #...
```

__2.2: Activation Functions and Derivatives__  
Write the hyperbolic tan and sigmoid activation functions, followed by their derivatives:
```python
def tanh(z):
    return #... 

def sigmoid(z):
    return #... 

def sigmoid_prime(z):
    return #... 

def tanh_prime(z):
    return #... 
```

__2.3: Cross-entropy cost function__  
Write the cross-entropy cost function.  
_Hint: It was done in assignment 1_  
```python
def cross_entropy_cost(y_preds, y_vals):
    #... 
    return #... 
```

__2.4: Derivative of the Loss__  
As seen in class, the loss function for classification is defined as:
\begin{equation*}
L\left( \hat{y}^{(i)}, y^{(i)}\right) = - \left[ \;\; y^{(i)} \log\left( \hat{y}^{(i)} \right) \;\;+\;\; \left(1- y^{(i)}\right) \log\left( 1 - \hat{y}^{(i)} \right) \;\;\right]
\end{equation*}

Find the expression of:
\begin{equation*}
\frac{\mathrm{d} }{\mathrm{d} \hat{y}^{(i)}} L\left( \hat{y}^{(i)}, y^{(i)}\right) = \cdots
\end{equation*} 
Use a text cell and LaTeX. If you are new to LaTeX, this [application](https://editor.codecogs.com/) will help.

Once you have an expression for the derivative of the loss, code it in python:
```python
def L_prime(y_preds, y_obs):
    return #...
```

## 3. Feedforward
It is time to write the feedforward function! Some information about the network:
* The network has two hidden layers 
* The nodes of the hidden layers use the hyperbolic tangent as activation function
* The output layer uses the sigmoid function 

__3.1: Feedforward Propagation__  
We do not have to know the number of activation units in each layer as the code here is made to be general. The weights and biases are given as input, stored in lists. Again, you do not have to worry about the indices yet (and I have simplified it by reshaping the input data for you). 

Complete the function below that does the feedforward propagation of the network, using the equations shown in class. Look at the output to see how to name your variables!

```python
def feedforward(input_X, weights, biases):

  W1, W2, W3 = weights ; b1, b2, b3 = biases

  m  = len(input_X) 
  a0 = input_X.reshape((m, 1, -1))

  # First layer
  #...
  #...

  # Second layer
  #...
  #...

  # Third layer
  #...
  #...
  
  nodes = [a0, z1, a1, z2, a2, z3, a3]

  return nodes
```

```{admonition} Check Point
:class: note
At that point in the assignment, it is possible to ask for a checkup with the instructor before moving _forward_ to ...the _backward_ pass (pun intended).
Send an email to the instructor with the exact title "[Machine Learning Course] Assignment 3 Check 1" where you paste the code of the function above (do not send any notebook).
```

__3.2: Predict__  
`````{margin}
To insert a code block within a text cell in Jupyter notebook, write:
````
   ```python
   # your code here

   ```
````
`````
This function is given (at the Setup step), yet there is a question for you: 
* What is the `output_node` in the context of our 2-hidden-layered neural network? 
* What type of values does the function `predict` return? 
* After successfully executing the `feedforward` function, how would you call the function `predict`? 


## 4. Neural Network Training
Time to train! The code below is a skeleton. You will have to complete it. To help you further, hyperparameters are given. You should replace the `#...` with your own code.


```python
# Hyperparameters
alpha = 0.03
N = 1000 # epochs

# Initialization 
m = len(X_train)     # number of data samples
n = X_train.shape[1] # number of input features
q = 3 # number of nodes in first hidden layer
r = 2 # number of nodes in second hidden layer

# WEIGHT MATRICES + BIASES
W1 =  #... 
W2 =  #... 
W3 =  #... 
b1 =  #... 
b2 =  #... 
b3 =  #... 

# OUTPUT LAYER
y_train = np.reshape(y_train, (-1, 1, 1))
y_test  = np.reshape(y_test , (-1, 1, 1))

# Storing cost values for train and test datasets
costs_train = []
costs_test  = []
debug = False

print("Starting the training\n")

# -------------------
#   Start iterations
# -------------------
for t in range(1, N+1):

  # FORWARD PROPAGATION
  # Feedforward on test data:
  nodes_test =  #... 
  ypreds_test =  #... 

  # Feedforward on train data:
  a0, z1, a1, z2, a2, z3, a3 = #... 
  ypreds_train =  #... 

  # Cost computation and storage
  J_train = cross_entropy_cost(ypreds_train, y_train)
  J_test  = cross_entropy_cost(ypreds_test,  y_test )
  costs_train.append(J_train)
  costs_test.append(J_test)

  if (t<=100 and t % 10 == 0) or (t>100 and t % 100 == 0):
      print(f"Iteration {t}\t Train cost = {J_train:.4f}  Test cost = {J_test:.4f}   Diff = {J_test-J_train:.5f}")

  # BACKWARD PROPAGATION
  # Errors delta:
  delta_3 = #...
  delta_2 = #...
  delta_1 = #...
  
  # Partial derivatives:
  dCostdW3 = #...
  dCostdW2 = #...
  dCostdW1 = #...
  dCostdb3 = #...
  dCostdb2 = #...
  dCostdb1 = #...

  if debug and t<3:
    print(f"a0: {a0.shape} a1: {a1.shape} a2: {a2.shape} a3: {a3.shape} ")
    print(f"W3: {W3.shape} z1: {z1.shape} z2: {z2.shape} z3: {z3.shape} ")
    print(f"dCostdW3: {dCostdW3.shape} dCostdW2: {dCostdW2.shape} dCostdW1: {dCostdW1.shape}") 

  # Update of weights and biases
  W3 = #... 
  W2 = #... 
  W1 = #... 
  b3 = #...  
  b2 = #...  
  b1 = #... 

print(f'\nEnd of gradient descent after {t} iterations')
```

````{admonition} Hints (lots of)
:class: tip, dropdown
__Hints on weight initialization__  

The weight matrices and bias vectors can be simply initialized with a random number between 0 and 1. Use:
```python
an_i_by_j_matrix = np.random.random(( i , j ))
```
The double parenthesis is important to get the correct shape. To make your code modular, use the variables encoding the number of activation nodes on the first and second hidden layers.

__Feedforward and predict__  
* To return the last element of a list: `my_list[-1]`

__Dimension of Nodes__  
At the end of the lecture on {ref}`NN2:backprop`, the equations are written with the indices. In python, we will use 3D NumPy arrays of shape ($i$, $j$, $k$), with 
* $i$ the sample index
* $j$ the number of rows of the node vector
* $k$ the number of columns

__Matrix/Vector Operations__  
* In python the matrix multiplication is done using `@`
* The element-wise multiplication is done using `*`

__Transpose__  
* To transpose a matrix M: `M.T` 
* To transpose 3D arrays, the indices to reorder are indicated as argument: `np.transpose(my3Darray, (0, 2, 1))` will transpose the two last dimensions.

__Summing__  
* To sum a 3D array on the first index: `np.sum(my3Darray, axis=0)`
````

## 5. Plots

__5.1: Cost evolution__  
Call the provided function to plot the cost evolution of both the training and testing sets.

__5.2: Scatter Plot__  
Use the `get_decision_surface` and `plot_scatter` functions to visualize the decision boundaries of your trained neural network. Did your neural network successfully learn the XOR function?

If all goes well, you should obtain something like this:

```{figure} ../images/a03_data_NN.png
---
  name: a03_data_NN
  width: 90%
---
Scatter plot of data representing the XOR configuration and the neural network performance.
```



```{warning}
&nbsp;  
This assignment is individual. The instructor will be available to answer questions per email and/or during special session. If you are stuck, it is possible to get some elements of answer (at the cost of some mark, but not much) __from the instructor only__.
```

```{important}
&nbsp;  
You can use the internet such as the official pages of relevant libraries, forum websites to debug, etc. You can use ChatGPT to help you locally with any python difficulties, but you cannot submit this entire assignment to the bot (and that will not even work).

The instructor and tutors are available throughout the week to answer your questions. Write an email with your well-articulated question(s). The more specific you are in your request, the best we can help you.

Thank you and do not forget to have fun while coding!
```