(linReg:lr)=
# Learning Rate

The learning rate $\alpha$ is a hyperparameter intervening in the calculation of the step size at which the parameters will be incremented or decremented. It controls the speed at which the algorithm 'learns.' Or said more mathematically, it sets how fast -- or slow -- we want the gradient descent to converge to the minimum cost.

## Learning Rate and Convergence
The learning rate is not directly setting the step size. It is a coefficient. With a fixed $\alpha$, the gradient descent can converge as the steps will become smaller and smaller due to the fact that the derivatives $\frac{\partial }{\partial \theta_j} C(\boldsymbol{\theta})$ will get smaller (caution: in absolute value!) as much as we approach the minimum:

````{margin}
Recall that the step size is given by $-\alpha \frac{d}{d \theta} C(\boldsymbol{\theta}) = -\alpha \times$ _slope_. Here the magnitude of the slope is decreasing at each iteration. 
````
```{figure} ../images/linReg_smaller_steps.png
---
  name: linReg_smaller_steps
  width: 80%
---
: The step size is reduced at the next iteration of the gradient descent,  
even if $\alpha$ remains constant.  
 <sub>Image from the author</sub>
 ```

## Learning Rate and Divergence
A learning rate too big will generate an updated parameter on the other side of the slope.  
Two cases:
* __The zig-zag__: if the next parameter $\theta'$ is at a smaller distance to the $\theta^{\min C}$ minimizing the cost function ($ | \theta' - \theta^{\min C} |  < | \theta - \theta^{\min C} |$), the gradient descent will generate parameters oscillating on each side of the slope until convergence. It will converge, but it will require a lot more steps. 
* __Divergence__: if the next parameter is at a greater distance than the $\theta^{\min C}$ minimizing the cost function ($ | \theta' - \theta^{\min C} |  > | \theta - \theta^{\min C} |$), the gradient descent will produce new parameters further and further away, escaping the parabola! It will diverge. We want to avoid this.

The divergence is illustrated on the right in the figure below: 
```{figure} ../images/linReg_learningRate_small_big.jpg
---
  name: linReg _learningRate_small_big
  width: 100%
---
: The learning rate determines the step at which the parameters will be updated (left). Small enough: the gradient descent will converge (middle). If too large, the overshoot can lead to a diverging gradient, no more "descending" towards the minimum (right).  
<sub>Image from [kjronline.org](https://www.kjronline.org/ViewImage.php?Type=F&aid=658625&id=F7&afn=68_KJR_21_1_33&fn=kjr-21-33-g007_0068KJR)</sub>
 ```


## Summary
* The learning rate $\alpha$ is a hyperparameter intervening in the calculation of the step size at which the parameters will be incremented or decremented.
* The step size varies even with a constant $\alpha$ as it is multiplied by the slope, i.e. the derivatives of the cost function.
* A small learning rate is safe as it likely leads to convergence, yet too small values will necessitate a high number of epochs.
* A large learning rate can make the next update of parameters overshoot the minimum of the cost function and lead to either 
  * an oscillating trajectory: the algorithm converges yet more iterations are needed
  * a diverging path: the gradient descent fails to converge


```{admonition} Question
:class: seealso
How to choose the best value for the learning rate?
```
We will discuss your guesses. Then the next section will give you some tricks!