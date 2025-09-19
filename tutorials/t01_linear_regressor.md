# T1. Linear Regressor

In this tutorial, you will learn how to code a linear regressor in Python. Part I will be very guided, Part II a bit less and Part III even less.

```{admonition} Important Note
:class: warning
All of you are coming with different backgrounds and programming proficiencies. Even if you are an experienced coder, this in-depth tutorial can  give you some insights you did not have before if you are mostly used to handling advanced, automated libraries. Do all parts and if you are done and start getting bored, come talk to the instructor. I will give you extra challenges 😉.
```


```{admonition} Learning Objectives
:class: tip
* Read a data file and manipulate data in pandas dataframes and NumPy arrays  
* Write functions relevant to linear regression  
* Write unit tests to check the correct behaviour of those functions  
* Code a linear regressor with one input feature “by hand,” i.e. implementing the equations explicitly  
* Generalize the code to handle $n$ input features  
* Rewrite the code using an OOP approach by defining a class  
* Test the class with datasets of different dimensionalities  
* Compare the results of your linear regressor with NumPy tools such as `polyfit` or the least-square method `lstsq`  
```


Open a fresh Colaboratory file or a local Jupyter Notebook and let's go!


## Part I: Linear Regressor By Hand (guided)

### 1.1 Get the data

[Download dataset](https://drive.google.com/uc?export=download&id=19CPweswe31ifoYxdl88XCZDprC3F6md9)

Mount your Drive according to the {ref}`tuto:setup` section, or retrieve it from your local folders if you are using Jupyter Notebook on your device. 

Import your classic:

```python
import pandas as pd
import numpy as np
```

Load your dataset from the CSV file into a Pandas DataFrame. This will allow you to inspect the data, handle missing values, and manipulate columns easily before turning it into numerical arrays. 

```python
df = pd.read_csv(your_data_file)
df.head()
```

Dataframes are very handy to visualize the dataset. Feel free to practice on this. Now we want fast computations: we will create a Numpy array for each column.

```python
x = df["x"].to_numpy()
y = df["y"].to_numpy()
```
### 1.2 Plot the data
Use the first plotting macro from the {ref}`app:t1:snippet:zone` to plot the data. What is the trend?


### 1.3 Functions
We will have to make the same computations several times, for instance while calculating the hypothesis function. Thus it is better to define proper functions for that. Functions in programming are making a code reusable, versatile and easier to read.

__Hypothesis function__  
This one is given for a single input feature: 

````{margin}
If you want to anticipate the next section, you can write a more versatile function `h_linear(thetas, X)` that would be general for $n$ input features. But for this part, the provided function on the left will do the job.
````
```python
# Single feature linear hypothesis
def h_linear_single_feature(thetas, x):
    """
    Simple linear model for a single feature.

    Parameters
    ----------
    thetas : array-like
        [theta_0, theta_1] where theta_0 is the intercept and theta_1 is the slope.
    x : array-like
        Input feature, shape (m_samples,).

    Returns
    -------
    predictions : array
        Predicted values, shape (m_samples,).
    """
    return thetas[0] + thetas[1] * x
```
__Cost function__  
Your first mission is to write a function calculating the cost (Mean Square Error). 

```{admonition} Good Practice
:class: important
Before even starting to write code it is important to be very clear on what are the inputs, the main steps, the outputs. I recommend going back to the 'pen and paper' to first write the algorithm, list the different variables. Then it will make the coding much easier and less error-prone.
``` 

☝️ Once you have your function computing the cost, it is always a good practice to test it. 

Let’s take the values seen in class from the ideal dataset. We know the numbers already, so it will be handy. 
Complete the following, filling out the `...` placeholders:

```python
x_test = np.array([1, 2, 3, 4])
y_test = np.array([2, 4, 6, 8])

# Set intercept term to zero
theta_0 = 0

# List of values for theta_1 (the slope)
theta_1_candidates = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]

for ... in ...:
    # Call your cost function
    ... 
```

You should be able to print a list of costs with values identical to the ones you calculated during class.


__Residuals__  
There will be another handy function to code. Recall the derivative of the cost function. It contains the difference between the predicted value and the observed one, without squaring them. This difference has the technical name of “residual”.  
Write another function `get_residuals(...)` computing the residuals, which is common term in both partial derivatives of the cost, $\frac{\partial C(\boldsymbol{\theta})}{\partial \theta_0}$ and $\frac{\partial C(\boldsymbol{\theta})}{\partial \theta_1}$.

☝️ Test it! Try to write some code to make sure your function works and compute the right things.


### 1.4 Linear Regressor Loop
Let’s now get to the core of things: the regressor!

Some guidance:

* In the {ref}`app:t1:snippet:zone`, the function `should_print_iteration` can help you mitigate long terminal output while printing variables in your gradient descent loop. Take it! 🎁

* If you struggle with the convergence, get your instructor to give you hyperparameters and initial model parameters that definitely work. 

* Last but not least, skeleton of the code is given below to help you get started. 


```python
# Hyperparameters
alpha   = 
N       = 
epsilon =   # tolerance threshold on gradients; exit if abs(gradients) < epsilon 

# Initialization
theta_0 = 
theta_1 = 

# Check lists of x and y are of same length:
m = len(x)  # sample size
if m != len(y):
    raise ValueError("The lists of x and y values are not the same length!")

# Store parameter values for gradient descent visualizations
theta_0_grad_history = np.array([theta_0])
theta_1_grad_history = np.array([theta_1])

print("Starting gradient descent\n")

# -------------------
#  Start iterations
# -------------------
for iter_idx in range(N):

    #___________________________

    # Your code here


    #___________________________

    # Store thetas (for plotting)
    theta_0_grad_history = np.append(theta_0_grad_history, theta_0_new)
    theta_1_grad_history = np.append(theta_1_grad_history, theta_1_new)

    # Pretty print: every 10 iters until 100, then every 100 iters
    if should_print_iteration(iter_idx, 10, 100, 100):
        print(
            f"Iter {iter_idx:>4}\t"
            f"θ₀ = {theta_0_new:>7.3f}\t∂J/∂θ₀ = {grad_theta_0:>8.4f}\t"
            f"θ₁ = {theta_1_new:>7.3f}\t∂J/∂θ₁ = {grad_theta_1:>8.4f}\t"
            f"Cost = {cost:>8.5f}"
        )

    #___________________________

    # Your code here


    #___________________________

print(f"\nEnd of gradient descent after {iter_idx+1} iterations")
```

### 1.5 Visualize the gradient descent
We will reproduce the plots from {numref}`plot_linReg_3D` seen in class. The plotting macros are given in the {ref}`app:t1:snippet:zone`.  Copy paste 🎁 and see!

### 1.6 Let's check with NumPy
Let's compare with NumPy `polyfit`:
```python
slope, intercept = np.polyfit(x, y, deg=1)
```

Print the parameters of your manual linear regressor and the ones from NumPy `polyfit`. Are they the same? 


## Part II: Generalizing to $n$ features with vectorized gradient descent
We wrote our linear regressor for a single input feature, with two scalar parameters: the intercept and the slope. Let's generalize the code to handle multiple features by representing the parameters as a vector.

Adapt your linear regressor to use a parameter vector, enabling vectorized gradient descent.

Tips: 
* You can either adapt your `get_residuals` function for the vectorized case, or drop it entirely and compute the residuals directly in your loop when calculating the gradients.
* Print the shape of your variables to check that all operations have compatible dimensions. 
* You may have to rewrite the hypothesis and cost functions to make sure it works with your vectors and matrices (in particular if you change the dimension of your input feature matrix to accommodate the intercept term)

Have fun!

## Part III: Let's get classy with OOP
The goal of this section is to give an initial example for Object-Oriented Programming (OOP) and show how it can be very convenient.

We just coded a linear regressor in a generalized fashion with $n$ input features. However, we fitted our model by explicitly writing step-by-step code, and if we want to use this code again, we have to copy-paste the previous blocks. We want to have a more efficient way, an implementation of a regressor that we can use again and again on different datasets - with different parameters. This is when OOP comes in. We need to turn our linear regressor into a class.

If you are new to OOP, find a teammate knowledgeable in the topic. If you are proficient in OOP, use this as an opportunity to improve your teaching skills by helping peers.

You will design a class LinearRegressor.

Before even writing a line of code, take a piece of paper and think about the following:

* What would be the attributes?
* What would be the methods?
* For each method, what would be the arguments?

Once you have this in place, discuss with your peers on your design strategy. When you are all good, get into the coding and don't forget to test it along the way. Then have fun with the dataset we worked on. There is another dataset to explore here:  
[Download dataset 2 ](https://drive.google.com/uc?export=download&id=1E-anZ2OcfsHyAHvcYXgqbn7FmaZkYgKG)

If you successfully fit this one, you can do the bonus or ask the instructor to give you a higher dimensionality dataset.

Then to compare your fitted parameters with NumPy, call the least-square meethod:
```python
# Compare with Numpy least-square solution
params_numpy = np.linalg.lstsq(X_aug, y, rcond=None)[0]
```

You can see in the Snippet Zone at the end a function to print a neat table of parameters from both your linear regressor class and NumPy, so you can compare them easily.


Enjoy building your own linear regressor class and exploring with it!

## Bonus
With the second data file, how could we visualize the data and the fit?  
This part is intentionally left open-ended. Bring your ideas 💡 and try them out 💻 


(app:t1:snippet:zone)=
## Appendix: Snippet Zone

### Plotting the Data
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,4), dpi=120)
ax.scatter(x, y, s=10)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.tight_layout()
```

(t1:app:pretty:print)=
### Pretty Printing

```python 
def should_print_iteration(iter_index, first_step, second_step, index_changing_step):
    """
    Trigger a print if iter_index matches step rules:
    print every `first_step` iterations before `index_changing_step`,
    then every `second_step` iterations afterwards.
    """
    if first_step <= 0 or second_step <= 0:
        raise ValueError("Steps must be positive integers.")

    if iter_index == 0:
        return True  # Always print at the first iteration

    if iter_index <= index_changing_step:
        return iter_index % first_step == 0
    else:
        return iter_index % second_step == 0
```

### Visualizing Gradient Descent
You can use as is or wrap it in a function.

```python
# Grid for 2D parameter space:
theta_0_grid = np.linspace(0, 30, 50)
theta_1_grid = np.linspace(0, 6, 50)

# Z values of costs for the surface:
meshed_theta_0, meshed_theta_1 = np.meshgrid(theta_0_grid, theta_1_grid)
meshed_costs = np.zeros_like(meshed_theta_0)  # Costs array

for i in range(meshed_theta_0.shape[0]):
    for j in range(meshed_theta_0.shape[1]):
        meshed_costs[i, j] = cost_function_linear_regression(
            [meshed_theta_0[i, j], meshed_theta_1[i, j]], x, y
        )

# Gradient descent: 5 first params then every 10 epochs
intermediary_theta_0_vals = np.concatenate(
    (theta_0_grad_history[0:5], theta_0_grad_history[5::5]), axis=None
)
intermediary_theta_1_vals = np.concatenate(
    (theta_1_grad_history[0:5], theta_1_grad_history[5::5]), axis=None
)

# Cost for selected intermediary weights (one per GD step)
intermediary_grad_cost_history = np.array(
    [
        cost_function_linear_regression([t0, t1], x, y)
        for t0, t1 in zip(intermediary_theta_0_vals, intermediary_theta_1_vals)
    ]
)

plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure(figsize=plt.figaspect(0.45))  # 16,4

# ==========================
#     Contour plot
# ==========================
plt.rcParams.update(plt.rcParamsDefault)

# Custom cost levels, from 5, step of 5, 200
levs = range(5, 200, 5)

ax = fig.add_subplot(1, 2, 1)

# Contour of theta parameter space:
CS = ax.contour(meshed_theta_0, meshed_theta_1, meshed_costs, levs, linewidths=0.5)
ax.clabel(CS, CS.levels[0:10], inline=1, fontsize=10, fmt="%d")

# Add the intermediary thetas from gradient descent:
ax.plot(intermediary_theta_0_vals, intermediary_theta_1_vals, marker=".", c="r", lw=0.5)
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
plt.title("Cost Function $J(\\theta_0, \\theta_1)$", loc="left", fontsize=10)

# ==========================
#     3D plot
# ==========================
ax = fig.add_subplot(1, 2, 2, projection="3d")

# 3D surface of cost vs (theta  0, theta 1):
surf = ax.plot_surface(
    meshed_theta_0,
    meshed_theta_1,
    meshed_costs,
    cmap="viridis_r",
    linewidth=0.3,
    alpha=0.5,
    edgecolor="k",
)

# Path of intermediary thetas from gradient descent:
ax.plot(
    intermediary_theta_0_vals,
    intermediary_theta_1_vals,
    intermediary_grad_cost_history,
    c="r",
)

ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("$J(\\theta_0, \\theta_1)$", rotation=90)
ax.azim = 170
ax.elev = 30
ax.xaxis.set_rotate_label(False)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_rotate_label(False)
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_rotate_label(False)
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.show()
```


### Print Table to Compare Manual vs NumPy Fits
```python
method_names = ["Manual gradient descent", "NumPy least-squares"]
parameters_list = [MyLinReg2Features.parameters, params_numpy]

num_decimals = 5

# Dynamic width for method column
method_col_width = max(len(name) for name in method_names) + 2  # +2 for padding
num_col_width = 12  # width for numeric columns

# Header
header = ["Method", "Intercept"] + [f"W{i}" for i in range(1, len(parameters_list[0]))]
print(f"{header[0]:<{method_col_width}}", end="")
for h in header[1:]:
    print(f" | {h:>{num_col_width}}", end="")
print()

# Separator
total_width = method_col_width + len(header[1:]) * (num_col_width + 3)
print("-" * total_width)

# Rows
for method, params in zip(method_names, parameters_list):
    print(f"{method:<{method_col_width}}", end="")
    for p in params:
        print(f" | {p:>{num_col_width}.{num_decimals}f}", end="")
    print()
```