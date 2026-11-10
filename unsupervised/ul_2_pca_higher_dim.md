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

# PCA in higher dimensions

When our data is higher dimensional than 2D we can project the data along more than one line. We've established that if we were only projecting along one line we would use the unit eigenvector with the highest eigenvalue of our covariance matrix to project the data. What about when we can project along more than one dimension? {math}`\hat{\Sigma}`


````{admonition} placeholder
:class: tip

Suppose that

````

Now let's return to our 3D example from the start of this section. In the plot below we've overlaid the three principle components (the eigenvectors) scaled by their eigenvalues. Projecting the data onto two of these principle components projects the data into the plane spanned by them.


```{admonition} Exercise
:class: seealso
Without considering the principle components, how can we visually check which plane would be best to project onto?
```

````{admonition} Check your answers
:class: tip, dropdown

If we align our viewing direction with each plane's normal vector (i.e. look at it "dead-on" or with an orthogonal perspective), we can see how the data would look when orthogonally projected onto the plane. By observing how 'spread' the projected points are we can get an idea of the variance of that projection. Hence, through visual inspection alone, we could conclude that the yellow plane is best.
````


```{code-cell}
:tags: [remove-input]

import k3d
import numpy as np
from sklearn.decomposition import PCA

# --- Data Generation ---
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

data = np.vstack([x, y, z]).T.astype(np.float32)

# --- PCA Calculation ---
# 1. Center the data by subtracting the mean.
mean = np.mean(data, axis=0)

# 2. Fit PCA to find all three principal components (eigenvectors).
pca = PCA(n_components=3)
pca.fit(data)

# 3. The principal components are the eigenvectors of the covariance matrix.
pc1, pc2, pc3 = pca.components_

# --- K3D Visualization ---
# 1. Initialize the plot
plot = k3d.plot(name="PCA Eigenvector Planes")

# 2. Set plot-wide attributes for a cleaner look
plot.background_color = 0xffffff
plot.grid_visible = True
plot.axes_helper = 0.0
plot.ticks_nb_x = 6
plot.ticks_nb_y = 6
plot.ticks_nb_z = 6
plot.axes = ['x', 'y', 'z']
plot.label_color = 0x666666
plot.grid_color = 0xcccccc

# 3. Add the original 3D scatter plot
plot += k3d.points(
    positions=data,
    point_size=0.5,
    color=0x0000ff,
    name="Original Data"
)

# 4. Add the principal component vectors (eigenvectors) with an intuitive color scheme
# PC1: Red, PC2: Green, PC3: Blue
plot += k3d.vectors(
    origins=np.array([mean, mean, mean]),
    vectors=pca.components_ * np.sqrt(pca.explained_variance_)[:, np.newaxis] * 3,
    colors=[0xff0000, 0x00ff00, 0x0000ff, 0xff0000, 0x00ff00, 0x0000ff],
    labels=["PC1", "PC2", "PC3"],
    label_size=0.5
)

# --- Plane Generation (using k3d.mesh for robustness) ---

def get_plane_mesh(center, v1, v2, size=20, rows=10, cols=10):
    """Generates vertices and triangle indices for a plane mesh."""
    # Create a grid of coefficients for the basis vectors
    coeffs = np.linspace(-size / 2, size / 2, rows)
    grid_x, grid_y = np.meshgrid(coeffs, coeffs)
    
    # Generate vertices using the parametric equation of a plane
    vertices = center + grid_x[..., np.newaxis] * v1 + grid_y[..., np.newaxis] * v2
    vertices = vertices.reshape(-1, 3).astype(np.float32)
    
    # Generate triangle indices to connect the vertices
    indices = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Get the four corners of a quad in the grid
            i0 = r * cols + c
            i1 = i0 + 1
            i2 = i0 + cols
            i3 = i2 + 1
            # Create two triangles from the quad
            indices.extend([[i0, i1, i2], [i1, i3, i2]])
            
    return vertices, np.array(indices, dtype=np.uint32)

# Plane 1: Spanned by PC1 (Red) and PC2 (Green) -> Yellow Plane
vertices1, indices1 = get_plane_mesh(mean, pc1, pc2)
plot += k3d.mesh(
    vertices1, indices1,
    color=0xFFFF00, opacity=0.3, name="Plane (PC1, PC2)"
)

# Plane 2: Spanned by PC1 (Red) and PC3 (Blue) -> Magenta Plane
vertices2, indices2 = get_plane_mesh(mean, pc1, pc3)
plot += k3d.mesh(
    vertices2, indices2,
    color=0xFF00FF, opacity=0.3, name="Plane (PC1, PC3)"
)

# Plane 3: Spanned by PC2 (Green) and PC3 (Blue) -> Cyan Plane
vertices3, indices3 = get_plane_mesh(mean, pc2, pc3)
plot += k3d.mesh(
    vertices3, indices3,
    color=0x00FFFF, opacity=0.3, name="Plane (PC2, PC3)"
)

# 5. Display the plot
plot.display()
```

 As we would expect, the plane which fits the data best is the plane spanned by the principle components with the greatest eigenvalues.
