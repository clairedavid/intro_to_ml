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





# Principle Component Analysis (PCA)
These next two methods are forms of *dimensionality reduction*, where we learn a lower-dimensional representation of our data. The lower dimensional space which we project our data into $\mathbb{R}^{L}$ is called a *latent* space and its elements are referred to as latent vectors.

<!-- ```{code-cell}
:tags: [remove-input]

import plotly.io as pio
pio.renderers.default = "notebook"


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


np.random.seed(42)
n_samples = 20
# Create a simple arc by reducing the range of the parameter 't'
t = np.linspace(0, 1.5 * np.pi, n_samples) # Creates a 3/4 circle arc
jitter = 0.5 # Controls the amount of random noise
radius = 8 # Controls the radius of the arc

# Create a simple arc using trigonometric functions and add random jitter
x = radius * np.cos(t) + np.random.randn(n_samples) * jitter
y = radius * np.sin(t) + np.random.randn(n_samples) * jitter
z = t * 2 + np.random.randn(n_samples) * jitter * 2 # Angling up with t

data = np.vstack([x, y, z]).T

# --- PCA Calculation ---
# 1. Center the data by subtracting the mean.
mean = np.mean(data, axis=0)
data_centered = data - mean

# 2. Fit PCA to find the principal components.
pca = PCA(n_components=2)
pca.fit(data) # Use original data; sklearn centers it internally for fitting.
data_projected_on_plane = pca.inverse_transform(pca.transform(data))

# 3. The principal components form the basis of the plane.
pc1, pc2 = pca.components_

# 4. The normal to the plane is the cross product of the components.
normal = np.cross(pc1, pc2)
# --- End of PCA Calculation ---


# Create a figure object.
fig = go.Figure()



# Add the original 3D scatter plot trace.
fig.add_trace(go.Scatter3d(
    x=data[:, 0],
    y=data[:, 1],
    z=data[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
    name='Original Data'
))

# Add the points projected onto the PCA plane.
fig.add_trace(go.Scatter3d(
    x=data_projected_on_plane[:, 0],
    y=data_projected_on_plane[:, 1],
    z=data_projected_on_plane[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='diamond'
    ),
    name='Projected Data'
))


# --- Code to visualize the PCA plane ---
# Create a meshgrid for the plane's surface.

x_range = np.linspace(-10, 10, 10)
y_range = np.linspace(-10, 10, 10)
X_plane, Y_plane = np.meshgrid(x_range, y_range)

# The plane equation is a*x + b*y + c*z + d = 0
# We can solve for z: z = (-a*x - b*y - d) / c
# The plane passes through the mean of the data.
a, b, c = normal
d = -np.dot(normal, mean)
Z_plane = (-a * X_plane - b * Y_plane - d) / c

# Add the plane surface trace.
fig.add_trace(go.Surface(
    x=X_plane,
    y=Y_plane,
    z=Z_plane,
    colorscale='greens',
    opacity=0.5,
    showscale=False,
    name='PCA Plane of Best Fit',
    showlegend=True,
))

# --- End of plane visualization code ---


# Update the layout for clarity and aesthetics.
fig.update_layout(
  title='',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=40) # Adjust margins
)

# Display the interactive plot.
fig.show()
```

 -->

```{code-cell}
:tags: [remove-input]

import k3d
import numpy as np
from sklearn.decomposition import PCA

# --- Dnp.random.seed(42)
n_samples = 20
# Create a simple arc by reducing the range of the parameter 't'
t = np.linspace(0, 1.5 * np.pi, n_samples) # Creates a 3/4 circle arc
jitter = 0.5 # Controls the amount of random noise
radius = 8 # Controls the radius of the arc

# Create a simple arc using trigonometric functions and add random jitter
x = radius * np.cos(t) + np.random.randn(n_samples) * jitter
y = radius * np.sin(t) + np.random.randn(n_samples) * jitter
z = t * 2 + np.random.randn(n_samples) * jitter * 2 # Angling up with t

data = np.vstack([x, y, z]).T.astype(np.float32)

# --- PCA Calculation ---
# 1. Center the data by subtracting the mean.
mean = np.mean(data, axis=0)
data_centered = data - mean

# 2. Fit PCA to find the principal components.
pca = PCA(n_components=2)
pca.fit(data) # Use original data; sklearn centers it internally for fitting.
data_projected_on_plane = pca.inverse_transform(pca.transform(data))

# 3. The principal components form the basis of the plane.
pc1, pc2 = pca.components_

# 4. The normal to the plane is the cross product of the components.
normal = np.cross(pc1, pc2)


# --- K3D Visualization ---
# 1. Initialize the plot
plot = k3d.plot(name="PCA Visualization")

# 2. Set plot-wide attributes for a cleaner look
plot.background_color = 0xffffff  # White background
plot.grid_visible = True         # Turn off the grid cube
plot.axes_helper = 0.0            # Turn off the small XYZ axes in the corner

plot.ticks_nb_x = 6
plot.ticks_nb_y = 6
plot.ticks_nb_z = 6

plot.axes =['x','y','z']


# FIX: Make grid and labels less prominent
plot.label_color =  0xffffff  #0x666666  # A dark grey for labels
plot.grid_color = 0xcccccc   # A light grey for the grid

# 3. Add the original 3D scatter plot with a name
plot += k3d.points(
    positions=data,
    point_size=0.5,
    color=0x0000ff,  # Blue
    name="Original Data" # This name will appear in the panel
)

# 4. Add the projected points with a name
plot += k3d.points(
    positions=data_projected_on_plane,
    point_size=0.5,
    color=0xff0000,  # Red
    name="Projected Data", # This name will appear in the panel
    shader="flat"
)

# 5. Add the PCA plane with a name and new color


# 1. Create a meshgrid for the plane's surface, based on the data's range
x_range = np.linspace(data[:, 0].min() - 2, data[:, 0].max() + 2, 10)
y_range = np.linspace(data[:, 1].min() - 2, data[:, 1].max() + 2, 10)
X_plane, Y_plane = np.meshgrid(x_range, y_range)

# 2. Calculate Z values using the plane equation: a*x + b*y + c*z + d = 0
a, b, c = normal
d = -np.dot(normal, mean)

# Solve for z: z = (-a*x - b*y - d) / c
# We add a very small number to c to avoid division by zero if the plane is vertical
Z_plane = (-a * X_plane - b * Y_plane - d) / (c + 1e-9)

# 3. Add the plane using k3d.surface
plot += k3d.surface(
    Z_plane.astype(np.float32),
    xmin=x_range.min(), xmax=x_range.max(),
    ymin=y_range.min(), ymax=y_range.max(),
    color=0xADD8E6,
    opacity=0.5,
    name="PCA Plane of Best Fit"
)

# 6. Display the plot, hiding the menu panel by default
#plot.menu_visibility = False
plot.display()
```

Principle Component Analysis (PCA) is a *nonparametric* form of dimensionality reduction where we *linearly* and *orthogonally* project the data $\mathcal{D}=\{x_n\mid1\leq n\leq N, x_n\in\mathbb{R}^{D}\}$ into a lower dimensional subspace $\mathcal{Z}$ ($\mathrm{dim}(\mathcal{Z})=L$) in such a way that the lower dimensional representation can be used to obtain a good reconstruction of the original data. This is achieved by finding the $L$ directions along which the data varies the most (the *principle components*) and projecting the data onto only these directions.


This projection is defined by a matrix $D\times L$ matrix $W$, the projection is given by $z=W^Tx$ and "unprojection" is given by $\hat{x}=Wz$. We measure the quality of our reconstruction by Mean-Squared Error:

$$
\mathcal{L}(W) = \frac{1}{N}\sum_{n=1}^{N}{\left\Vert x_n - W(W^Tx_n) \right\Vert}_{2}^2
$$

This objective can be minimised by setting $W=U_L$, where $U_L$ contains the $L$ eigenvectors with the largest eigenvalues of the empirical covariance matrix of the data $\hat{\Sigma}$:

$$
\hat{\Sigma} = \frac{1}{N}\sum_{n=1}^N(x_n-\hat{x})(x_n-\hat{x})^T
$$


In the above Figure, we can see an example of this; a 3D spiral has been projected onto the 2D plane from which the data could be best reconstructed. Why is this a useful thing to do? Sometimes dimensionality reduction is useful simply because it allows us to visualise how our data varies. In many real-world settings our datasets contain far more than 3 variables and thus qualitatively examining the data can be difficult. We could also apply other machine learning algorithms to the lower dimensional data for a computational speedup. This can be particularly effective when most of the variation in our data can be captured in far fewer dimensions than the data's dimensionality.

```{margin}
:class: warning

In the next section we will assume that you're comfortable with *eigenvalues* and *eigenvectors*. If you're feeling rusty try this [3Blue1Brown video](https://www.youtube.com/watch?v=PFDu9oVAE-g&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=14).
```

We'll now build up some intuition for why $U_L$ is the optimal orthogonal projection of the data. Let's do a walkthrough with some toy data.


