---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
mystnb:
  execution_mode: cache
  execution_timeout: 300
  execution_raise_on_error: true
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Final comments on PCA

## PCA in practice
In these notes we have presented PCA as involving an *eigendecomposition* of the covariance matrix, however, there are alternatives (which yield the same principle components) that may be preferrable in practice. Firstly, when the dimensionality of the data is high, the memory requirements of storing the covariance matrix can grow to be very large. In these cases, variants of the *Singular Value Decomposition* are preferred as they bypass the need to compute the covariance matrix (See the svd_solver section of [the scikit-learn docs on PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)). Another issue is the *scale* of the variables, since PCA gives us the optimal linear and orthogonal projection with respect to a *squared* reconstruction loss, if some variables have a much larger scale than others they will dominate the reconstruction loss, so PCA will overly favour those variables. For example, if you have data on cars and the weight is measured in grams but the length is measured in kilometers, then the weight will completely dominate PCA. It is therefore important to think about the scale of your variables and to rescale the data when appropriate. For more on PCA, including extensions of it, {cite:p}`a-pml1Book` is an excellent resource.


<!-- ```{code-cell}
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
import matplotlib.pyplot as plt

rng_gen = np.random.default_rng(32)

N = 200
P = 3
rho = 0.5


cov = np.array([[1.0, 0.8, 0.2],
                [0.8, 1.0, 0.1],
                [0.2, 0.1, 50.0]])
X = rng_gen.multivariate_normal(np.zeros(3),cov,size=N)

X=X-np.mean(X,axis=0)
X_cov = np.cov(X,rowvar=False)
eigenvalues,eigenvectors = np.linalg.eig(X_cov)
eigenvectors = 5* eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

#Sort and print
eval_and_evec = list(sorted(zip(eigenvalues, eigenvectors),key=lambda x: -x[0]))


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# --- 3. Plot the Data ---
# Create the scatter plot
ax.scatter(X[:,0], X[:,1], X[:,2], c='b', marker='o')

ax.quiver(0, 0, 0, eval_and_evec[0][1][0], eval_and_evec[0][1][1], eval_and_evec[0][1][2], color='r',label=r'$e_1$')
# --- 4. Customize and Show Plot ---
# Add labels for clarity
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot Example')

# Display the plot
plt.show()
``` -->



## The need for non-linear transformations
In the next section we'll be taking a look at *AutoEncoders*. These are neural networks which also perform dimensionality reduction, however, their projection into a lower dimensional space can be *non-linear*. To motivate why this is useful we'll quickly look at an example where linear projections struggle.


Consider a semicircular cluster of points with some jitter. We'll plot it with its eigenvectors.

```{code-cell}
import numpy as np
n_samples=50
theta = np.linspace(0, np.pi, n_samples)

x = np.cos(theta)
y = np.sin(theta)

jitter_x = 0.05 * np.random.randn(n_samples)
jitter_y = 0.05 * np.random.randn(n_samples)
semicircle=np.vstack((x + jitter_x, y + jitter_y)).T

#Center the data
centered_semicircle = semicircle - np.mean(semicircle,axis=0)
```

```{code-cell}
:tags: [remove-input]

import matplotlib.pyplot as plt
semi_cov = np.cov(centered_semicircle,rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(semi_cov)

#Normalise the eigenvectors
eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)
eigenvectors = eigenvectors[np.argsort(-eigenvalues)]

fig,ax = plt.subplots()
ax.scatter(centered_semicircle[:,0],centered_semicircle[:,1],zorder=4)

ax.quiver(0, 0, eigenvectors[0][0], eigenvectors[0][1],
           angles='xy', scale_units='xy', scale=1, color='r', width=0.01,label=r'$e_1$',zorder=5)
ax.quiver(0, 0, eigenvectors[1][0], eigenvectors[1][1],
           angles='xy', scale_units='xy', scale=1, color='b', width=0.01,label=r'$e_2$',zorder=5)
ax.grid(True)
ax.set_ylim(-1.25,1.25)
ax.set_xlim(-1.25,1.25)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
```

Clearly, projecting along either eigenvector will lead to poor reconstruction of the data.
However, since this data is a perturbed semicircle, if we first map the points to their polar coordinates (a non-linear transformation), the data will much better suited to PCA!

```{code-cell}
#Polar coordinates of uncentered data
Rs = np.linalg.norm(semicircle,axis=1)
thetas = np.arctan(semicircle[:,1]/semicircle[:,0])

#Center the data
Rs = Rs-np.mean(Rs,axis=0)
thetas = thetas -np.mean(thetas,axis=0)
polar_data = np.stack((Rs,thetas),axis=1)

#Eigendecomp of cov matrix
polar_cov = np.cov(polar_data,rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(polar_cov)

#Normalise the eigenvectors
eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

eigenvectors = eigenvectors[np.argsort(-eigenvalues)]
```

```{code-cell}
:tags: [remove-input]

fig,ax = plt.subplots()

ax.quiver(0, 0, eigenvectors[0][0], eigenvectors[0][1],
           angles='xy', scale_units='xy', scale=1, color='r', width=0.01,label=r'$e_1$',zorder=5)
ax.quiver(0, 0, eigenvectors[1][0], eigenvectors[1][1],
           angles='xy', scale_units='xy', scale=1, color='b', width=0.01,label=r'$e_2$',zorder=5)

ax.scatter(Rs, thetas,zorder=4)
ax.grid(True)
ax.set_ylim(-2,2)
ax.set_xlim(-2,2)
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\theta$')
plt.show()
```

Hopefully this convinces you that *non-linear* transformations can indeed be very useful!

```{bibliography}
:filter: docname in docnames
:keyprefix: a-
```
