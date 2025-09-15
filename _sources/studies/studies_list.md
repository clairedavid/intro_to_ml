# List of STEPs

````{margin}
Difficult STEP are highlighted with a 🧠 emoji.
````

## Linear Regressor’s Stability
Find visuals to depict the zones of divergence of the linear regressor (in 1D) with respect to the choice of initial random parameters and learning rate.

## Feature Ranking with Different Classifiers
Is feature importance absolute, or relative to a given classifier? Explore this by comparing how different classifiers or regressors rank features.

## Support Vector Machine  
For several datasets, compare the performance of an SVM vs a Random Forest classifiers (if time, add a neural network).

## AdaBoost by Hand 🧠 
Implement a version of the AdaBoost algorithm by hand and compare it with Scikit-Learn. Choose the relevant metrics and visuals. 
Bonus: find the error in Scikit-Learn in the calculation of the probability!

## GradientBoost by Hand 🧠🧠
Implement a version of the Gradient Boosting algorithm and compare it with the state-of-the-art XGBoost.

## Padding vs Not Padding  
Is a neural network performing the same (accuracy & speed) if the computations are split between weights and biases or if the weight matrix is padded with an extra row/column to include the biases?

## Manual implementation of momentum
Showcase the effect of adding momentum in a neural network. Pick a relevant dataset and, as much as possible, illustrate it with clear visuals.

## Optimizers by Hand 🧠
Implement the update rules of RMSProp and/or Adam, and illustrate how they differ from vanilla SGD.
Extension: visualize the trajectory of parameters on a simple loss surface to highlight the differences.

## Compare Adam with L-BFGS Method
Study the rate of convergence of Adam vs L-BFGS. Is there some virtue of starting off with a second order method and switch to a first order one?
Note: you are not required to implement them manually.  

## Neural Network Initialization  
Compare the performance (and possible cases of divergence) of a neural network with different initialization schemes. 
Extension: display the variance at each layer and ideally, how the variance evolves during training.

## Stochastic vs Batch 
Showcase the pros & cons of the Stochastic Gradient Descent vs Batch Gradient Descent by varying the number of features, sample size and complexity of the data set.

## Clustering++
There are so many clustering algorithms. Pick your favourite and explain their pros & cons. With a code.

## Manual CNN vs PyTorch CNN 🧠
Build a Convolutional Neural Network by hand. Build the same with PyTorch. Work out the exact number of trainable parameters in your manual CNN. Compare with the one on PyTorch. They should match.
Bonus: initialize the same both networks and check the computations are identical. Good luck.
