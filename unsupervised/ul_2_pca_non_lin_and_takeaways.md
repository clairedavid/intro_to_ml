---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Further comments on PCA



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
