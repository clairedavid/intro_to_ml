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

# PCA Walkthrough with toy data
```{math}
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}
\newcommand{\var}{\mathrm{Var}}
\newcommand{\xvec}{\mathbf{x}}
\newcommand{\zvec}{\mathbf{z}}
\newcommand{\uvec}{\mathbf{u}}
\newcommand{\evec}{\mathbf{e}}
\DeclareMathOperator*{\exptop}{\mathbb{E}}
\newcommand{\exptx}{\exptop_{\xvec}}
```


```{code-cell}
import matplotlib.pyplot as plt
import numpy as np

# Sample data
toy_x_data = np.array([1, 2, 3],dtype=np.float32)
toy_y_data = np.array([2, 1.8, 4.5],dtype=np.float32)

#Center the data to prepare for PCA
toy_x_data = toy_x_data-np.mean(toy_x_data)
toy_y_data = toy_y_data-np.mean(toy_y_data)
toy_data = np.stack((toy_x_data,toy_y_data),axis=1)


#Plot the centered data
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(toy_x_data, toy_y_data, marker='o')
ax.grid(True)
ax.spines[['top','right']].set_visible(False)
ax.set_ylabel('Y')
ax.set_xlabel('X')

plt.show()
```

If we only consider the directions of the $x$-axis and $y$-axis, clearly the data varies more along the $y$-axis. Therefore, a projection onto the $y$-axis would better preserve the data! Thankfully, we are not only limited to these two directions. The widget below allows you try different directions to project the data onto by varying the angle of the projection line from the $x$-axis. 

```{admonition} Exercise
:class: seealso

Try to find the best projection line!

```



```{code-cell}
:tags: [remove-input]

import altair as alt
import pandas as pd
import numpy as np

# Embed the interactive chart in the HTML site.
alt.renderers.enable("html")

# Disable the 5000 row limit for this session.
alt.data_transformers.disable_max_rows()



# --- 2. Pre-calculate all data for all slider steps ---
angles = np.arange(0, 180.5, 2.0)
all_data = []
point_ids = range(len(toy_data))

for angle_deg in angles:
    angle_rad = np.deg2rad(angle_deg)
    u = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    projected_scalars = toy_data.dot(u)
    projected_data = np.outer(projected_scalars, u)
    variance = np.var(projected_scalars)
    reconstruction_loss = np.mean(np.square(projected_data - toy_data))
    
    for i in point_ids:
        all_data.append({
            'angle': angle_deg,
            'point_id': i,
            'x_orig': toy_data[i, 0],
            'y_orig': toy_data[i, 1],
            'x_proj': projected_data[i, 0],
            'y_proj': projected_data[i, 1],
            'variance': variance,
            'reconstruction_loss': reconstruction_loss
        })

df = pd.DataFrame(all_data)

# Create DataFrame for connecting lines
df_orig = df[['angle', 'point_id', 'x_orig', 'y_orig']].copy()
df_orig.rename(columns={'x_orig': 'x', 'y_orig': 'y'}, inplace=True)
df_proj = df[['angle', 'point_id', 'x_proj', 'y_proj']].copy()
df_proj.rename(columns={'x_proj': 'x', 'y_proj': 'y'}, inplace=True)
df_lines = pd.concat([df_orig, df_proj])

# --- 3. Build the Interactive Altair Chart ---
slider = alt.binding_range(min=0, max=180, step=2, name='Angle (°):')
angle_param = alt.param(name="angle_selector", value=0, bind=slider)

original_points = alt.Chart(df.drop_duplicates(subset='point_id')).mark_point(
    size=80, color='blue', opacity=0.3
).encode(
    x=alt.X('x_orig:Q', title='X', scale=alt.Scale(domain=[-2.5, 2.5])),
    y=alt.Y('y_orig:Q', title='Y', scale=alt.Scale(domain=[-2.5, 2.5])),
    tooltip=['point_id']
)

base_dynamic = alt.Chart(df).properties(width=600, height=500)

projected_points = base_dynamic.mark_point(
    size=80, color='orange', filled=True, shape='cross'
).encode(
    x='x_proj:Q',
    y='y_proj:Q',
    tooltip=[alt.Tooltip('variance:Q', format='.3f'), alt.Tooltip('reconstruction_loss:Q', format='.3f')]
).add_params(
    angle_param
).transform_filter(
    alt.datum.angle == angle_param
)

connecting_lines = alt.Chart(df_lines).mark_line(color='gray', strokeDash=[3,3]).encode(
    x='x:Q',
    y='y:Q',
    detail='point_id:N'
).transform_filter(
    alt.datum.angle == angle_param
)

df['x_line_start'] = -3 * np.cos(np.deg2rad(df['angle']))
df['y_line_start'] = -3 * np.sin(np.deg2rad(df['angle']))
df['x_line_end'] = 3 * np.cos(np.deg2rad(df['angle']))
df['y_line_end'] = 3 * np.sin(np.deg2rad(df['angle']))

projection_line = base_dynamic.mark_rule(color='red', size=2).encode(
    x='x_line_start:Q',
    y='y_line_start:Q',
    x2='x_line_end:Q',
    y2='y_line_end:Q'
).transform_filter(
    alt.datum.angle == angle_param
)

# --- CORRECTED SECTION ---
text_data = df[['angle', 'variance', 'reconstruction_loss']].drop_duplicates()

# 1. Define the text formatting as a Vega expression string
text_expression = (
    "'Variance: ' + format(datum.variance, '.3f') + "
    "'\\n' + "  # Use '\\n' for a new line in the expression
    "'Reconstruction Loss: ' + format(datum.reconstruction_loss, '.3f')"
)

# 2. Build the text layer using transform_calculate
text_layer = alt.Chart(text_data).mark_text(
    align='left', baseline='top', dx=10, dy=10, size=14, lineBreak='\\n'
).transform_calculate(
    # Create a new field named 'label' using the expression
    label=text_expression
).encode(
    # 3. Use the newly created 'label' field for the text encoding
    text='label:N'
).transform_filter(
    alt.datum.angle == angle_param
)
# --- END CORRECTED SECTION ---

chart = original_points + connecting_lines + projected_points + projection_line + text_layer

chart
```

````{margin}
For our purposes, 
```{math}
\exptx[\dots]=\frac{1}{N}\sum_{i=1}^{N}\dots
```
````

```{margin}
Here we're making the stronger assumption that the projection matrix's columns are *orthonormal* (orthogonal and unit). This will make the derivation easier and we will anyway be forced to assume that the principle components are unit later on.
```
````{admonition} Mathematical connection between the loss and the variance
:class: tip

What you've (hopefully) noticed is that higher variance in the projected data corresponds to lower reconstruction loss! This is why we choose project our data onto the directions with highest variance. Below we'll give a formal explanation for why this is the case:

Assume that our data is centered. Let $W$ be our $D\times L$ projection matrix with *orthonormal* columns.



Then {math}`\exptx[W^T\xvec]=W^T\exptx[\xvec]=0`. Hence, the variance of the projected data is
```{math}
\begin{align}
\var[W^T\xvec]&=\exptx\left[\left(W^T\xvec\right)^2\right]-\left(\exptx\left[W^T\xvec\right]\right)^2\\
&=\exptx\left[\left(W^T\xvec\right)^2\right]
\end{align}
```

Next, consider the reconstruction loss

```{math}
\begin{align}
\mathcal{L}(W)&=\exptx\left[(\xvec-WW^T\xvec)^2\right]\\
&=\exptx\left[\xvec^T\xvec - 2\xvec^TWW^T\xvec + \xvec^T W W^T W W^T\xvec \right]\\
&=\exptx\left[\xvec^T\xvec - 2\xvec^TWW^T\xvec + \xvec^T W W^T\xvec \right]\,\text{since } W^TW=\mathbf{I}\\
&=\exptx[\xvec^T\xvec - \xvec^T W W^T \xvec]\\
&=\exptx[\xvec^T\xvec] - \exptx[(W^T\xvec)^2]\\
\end{align}
```

Hence,

```{math}

\begin{align}
\argmax_{W}\,\var[W^T\xvec]&=\argmax_{W}\,-\mathcal{L}(W)+\exptx[\xvec^T\xvec]\\
&=\argmax_{W}\,-\mathcal{L}(W)\\
&=\argmin_{W}\,\mathcal{L}(W)
\end{align}
```

This means that if our projection is orthogonal, maximising the variance of the projected data is equivalent to minimising the reconstruction loss.
````

So how do we algorithmically find those directions? Well, it turns out that the directions (or lines) of greatest variation are the *eigenvectors* of the (empirical) covariance matrix of our data! Therefore, PCA amounts to simply finding these eigenvectors and projecting onto (some) of them.


````{admonition} Connections between the lines of greatest variation and their variance with the eigenvectors and their eigenvalues
:class: tip

Continuing from $\var[W^T\xvec]=\exptx[(W^T\xvec)^2]$ we can obtain

```{math}
\begin{align}
\var[W^T\xvec]&=\exptx[(W^T\xvec)^2]\\
&=\exptx[\xvec^TW W^T \xvec]\\
&=\exptx\left[\sum_{l=1}^{L}\xvec^{T}\uvec_{l}\uvec_{l}^T\xvec \right]\text{ where } \uvec_{l} \text{ is the } l^{\text{th}} \text{ column of } W\\ 
&=\sum_{l=1}^{L}\exptx\left[\xvec^{T}\uvec_{l}\uvec_{l}^T\xvec \right]
\end{align}
```

Since the latent space for our toy data will be 1D we'll focus on $L=1$ for now. This corresponds to finding a single vector $\uvec$ that maximises the variance of the projected data. Assume again that our data is centeted:

```{math}
\begin{align}
\var[W^T\xvec]&=\exptx[\xvec^{T}\uvec\uvec^T\xvec]\\
&=\exptx[\uvec^T\xvec\xvec^T\uvec]\\
&=\uvec^T\exptx[\xvec\xvec^T]\uvec\\
&=\uvec^T\hat{\Sigma}\uvec \text{, by the definition of the empirical covariance matrix for centered data}
\end{align}
```
Therefore, to maximise the variance of the projected data we should find the $\uvec$ that maximises $\uvec^T\hat{\Sigma}\uvec$. However, this is a poorly defined objective, for any proposed solution vector $\mathbf{v}$, the vector $2\mathbf{v}$ will produce a higher variance! Furthermore, what we really care about is the direction of this vector and from a practical standpoint we don't want our projections to be arbitrarily large. So we will impose the constraint that $\uvec$ should be a unit vector (i.e. $\uvec^T\uvec=1$). 

This is a constrained optimisation problem:
```{math}
\argmax_{\uvec, \Vert \uvec\Vert=1}\; \uvec^T\hat{\Sigma}\uvec
```
Let $f(\uvec)=\uvec^T\hat{\Sigma}\uvec$ and $g(\uvec)=\uvec^T\uvec-1$. Then $\nabla_{\uvec} f(\uvec)=2\hat{\Sigma}\uvec$ and $\nabla_{\uvec} g(\uvec)=2\uvec$. Hence, both of these functions are *continuously differentiable*. This suggests that we can use the method of Lagrange Multipliers to find candidate solutions for $\uvec$. Furthermore, since $\nabla_{\uvec} g(\uvec) \neq 0$ for all $\uvec$ satisfying $\uvec^T\uvec=1$, the method of Lagrange Multipliers will find **all** the candidate solutions to the problem.

We define the Lagrangian function $L$ as 

```{math}
\begin{align}
&L(\uvec,\phi) = \uvec^T\hat{\Sigma}\uvec - \phi(\uvec^T\uvec-1)\\
&\text{where } \phi \in \mathbb{R}
\end{align}
```
In order to find the candidate solutions we must solve the equations $\nabla_{\uvec}L=0$ and $\nabla_{\phi}L=0$.

First, we compute the gradients
```{math}
\begin{align}
&\nabla_{\uvec}L=2\hat{\Sigma}\uvec-2\phi\uvec \\
&\nabla_{\phi}L=1-\uvec^T\uvec
\end{align}
```

Setting $\nabla_{\uvec}L=0$, we obtain:
```{math}
\begin{align}
&2\hat{\Sigma}\uvec - 2\phi \uvec =0\\
&\hat{\Sigma}\uvec = \phi \uvec
\end{align}
```
This is precisely the definition for an eigenvector of $\hat{\Sigma}$ with eigenvalue $\phi$! Therefore our system of equations is satisfied by *unit* eigenvectors of $\hat{\Sigma}$. So the unit vector which maximises the variance of the projected data (or equivalently, minimises the loss) is amongst the unit eigenvectors of $\hat{\Sigma}$.

So, which one maximises the variance? It turns out that we can connect the variance of the projected data to each eigenvector's eigenvalue. Assume that the projection vector is given by a unit eigenvector $\evec$, then:

```{math}
\begin{align}
\var\left[\evec^T\xvec\right]&=\evec^T\hat{\Sigma}\evec\\
&= \evec^T\lambda \evec\text{, where } \lambda \text{ is } \evec\text{'s eigenvalue}\\
&=\lambda(\evec^T\evec)\\
&=\lambda
\end{align}
```

````

Ok! Let's take stock of what we've established
1. To reconstruct our data well we should maximise the variance of the projected data
2. The vectors which maximise the variance will be amongst the eigenvectors of our data's empirical covariance matrix
3. The eigenvalue associated with each *unit* eigenvector is equal to the variance of data projected onto that eigenvector

```{warning}
Our derivations have relied on the data being centered (zero mean). Some PCA fitting tools will implicitly center your data under the hood, but if you want to calculate the covariance matrix using $\frac{1}{N}\sum_{i=1}^{N}\xvec^{(i)}{\xvec^{(i)}}^T$, your data **must** be centered. In general, there is no downside to first centering your data.
```

Therefore we need to find the eigenvector(s) of our empirical covariance matrix with the highest eigenvalue(s)!

We can obtain our empirical covariance matrix easily with numpy:
```{code-cell}
toy_cov = np.cov(toy_data,rowvar=False)
print(toy_cov)
```

The eigenvalues of the empirical covariance matrix can be found by solving for the roots of $\hat{\Sigma}$'s *characteristic polynomial*:

$$
\mathrm{det}(\hat{\Sigma}-\lambda\mathbf{I})=0
$$

And for each eigenvalue $\lambda_i$ its corresponding eigenvectors are found by solving

$$
(\hat{\Sigma}-\lambda_i\mathbf{I})=0
$$



```{margin}
Assume that the eigenvalues are *distinct* (they have unique values). There are degenerate cases which complicate the first question but for most real-world datasets, the eigenvalues will be distinct.
```
```{admonition} Exercise
:class: seealso
1.) Is there more than one *unit* eigenvector *per eigenvalue* and if so, how many are there?<br/>
2.) Find the eigenvalues and *unit* eigenvectors for our toy example by hand. (Hint: When solving for the eigenvectors, the rows of the matrix should be multiples of each other)

```


````{admonition} Check your answers
:class: tip, dropdown

__Answer 1.__ Yes. There are exactly two unit eigenvectors per eigenvalue. They point in opposite directions along the same line.

__Answer 2.__ 
```{math}
\begin{align}
&\bullet \lambda_1\approx 3.032 \text{ and } e_1\approx\begin{pmatrix}0.524\\0.852 \end{pmatrix}\text{ or }\begin{pmatrix}-0.524\\-0.852 \end{pmatrix}\\
&\bullet \lambda_2\approx 0.231 \text{ and } e_2\approx\begin{pmatrix}-0.852\\0.524 \end{pmatrix}\text{ or }\begin{pmatrix}0.852\\-0.524\end{pmatrix}
\end{align}
```



````

<!-- ````{admonition} The unit eigenvectors and their eigenvalues
:class: tip, dropdown




```` -->

```{code-cell}
:tags: [remove-input]
e_1 = np.array([0.524, 0.852])
e_2 = np.array([-0.852, 0.524])
```

In order to map the data to a 1D subspace we need only perform a dot product between each datapoint and a vector:

```{code-cell}

e1_proj_1d = toy_data.dot(e_1)
e2_proj_1d = toy_data.dot(e_2)
```

Then to reconstruct the projected data we need only multiply the 1D data with the vector used to map it:

```{code-cell}
e1_un_proj = e_1*e1_proj_1d[:,None]
e2_un_proj = e_2*e2_proj_1d[:,None]
```

The best eigenvector to choose is the one with the highest eigenvalue! We can visualise all of this below. Make sure you understand this figure before proceeding, if you're unsure of something speak to a classmate, tutor or the lecturer.

```{code-cell}
:tags: [remove-input]

import matplotlib.lines as mlines
import matplotlib.patches as mpatches 
from matplotlib.legend_handler import HandlerTuple
# Define the individual vectors


# Origin point for the vectors
origin = np.array([np.mean(toy_x_data), np.mean(toy_y_data)])

# Create the plot
fig, axarr = plt.subplot_mosaic([['A','A','A','B','B'],['A','A','A','C','C']], figsize=(8,4))

#plt.subplots(figsize=(6,6))


# Plot each vector with a separate call to quiver
# This allows for individual styling and labeling
axarr['A'].quiver(origin[0], origin[1], e_1[0], e_1[1],
           angles='xy', scale_units='xy', scale=1, color='r', width=0.015,label=r'$e_1$')

axarr['A'].plot([-5*e_1[0],5*e_1[0]], [-5*e_1[1],5*e_1[1]], color='red', linestyle='--')
axarr['A'].quiver(origin[0], origin[1], e_2[0], e_2[1],
           angles='xy', scale_units='xy', scale=1, color='b', width=0.015,label=r'$e_2$')
axarr['A'].plot([-5*e_2[0],5*e_2[0]], [-5*e_2[1],5*e_2[1]], color='blue', linestyle='--')

axarr['A'].scatter(toy_x_data, toy_y_data, marker='o', s=60)
axarr['A'].scatter(e1_un_proj[:,0],e1_un_proj[:,1], marker='s', s=30, linewidth=3, zorder=5, color='crimson')
axarr['A'].scatter(e2_un_proj[:,0],e2_un_proj[:,1], marker='s',s=30, linewidth=3,zorder=5, color='dodgerblue')

axarr['A'].spines[['top','right']].set_visible(False)
# arr['A']Set plot limits and aspect ratio for an accurate representation
axarr['A'].set_xlim(-1.5, 2)
axarr['A'].set_ylim(-1.5, 2)
axarr['A'].set_aspect('equal', adjustable='box')

# arr['A']Add titles, labels, a grid, and a legend for clarity
#parr['A']lt.title("Visualization of Two Vectors (Separate Quiver Calls)")
axarr['A'].set_xlabel("X")
axarr['A'].set_ylabel("Y")
axarr['A'].grid(True)
#axarr['A'].legend()


axarr['B'].scatter(e1_proj_1d,np.zeros_like(e1_proj_1d),color='crimson',zorder=5,s=40,marker='D')
axarr['B'].axhline(0,color='red',linestyle='--')
#axarr['B'].spines[['top','right']].set_visible(False)
axarr['B'].set_yticks([])
axarr['B'].set_xlim(-3,5)
#axarr['B'].grid(True) 

axarr['C'].scatter(e2_proj_1d,np.zeros_like(e2_proj_1d),color='dodgerblue',zorder=5,s=40, marker='D')
axarr['C'].axhline(0,color='blue',linestyle='--')
#axarr['C'].spines[['top','right']].set_visible(False)
axarr['C'].set_yticks([])
axarr['C'].set_xlim(-3,5)
#axarr['C'].grid(True) 
# Display the plot

line_e1 = mlines.Line2D([], [], color='red', linestyle='-')
marker_e1 = mlines.Line2D([], [], color='red', marker='>', linestyle='None',
                          markersize=10, markerfacecolor='red')
handle_e1_arrow = (line_e1, marker_e1)

line_e2 = mlines.Line2D([], [], color='blue', linestyle='-')
marker_e2 = mlines.Line2D([], [], color='blue', marker='>', linestyle='None',
                          markersize=10, markerfacecolor='blue')
handle_e2_arrow = (line_e2, marker_e2)

# Dashed Lines
handle_e1_line = mlines.Line2D([], [], color='red', linestyle='--', label="e1's line")
handle_e2_line = mlines.Line2D([], [], color='blue', linestyle='--', label="e2's line")

# Simple Line for data
handle_data = mlines.Line2D([], [], color='tab:blue', linestyle='None', marker='o', label="data", markersize=8)

# Shapes (Squares and Diamonds)
handle_e1_recon = mlines.Line2D([], [], color='crimson', marker='s', linestyle='None',
                                markersize=8, label='e1 reconstruction')
handle_e1_proj = mlines.Line2D([], [], color='crimson', marker='D', linestyle='None',
                               markersize=8, label='e1 1D projection')

handle_e2_recon = mlines.Line2D([], [], color='dodgerblue', marker='s', linestyle='None',
                                markersize=8, label='e2 reconstruction')
handle_e2_proj = mlines.Line2D([], [], color='dodgerblue', marker='D', linestyle='None',
                               markersize=8, label='e2 1D projection')


# 4. Collect all handles and their corresponding labels
handles = [
    handle_e1_arrow,handle_e2_arrow, handle_data, handle_e1_line, handle_e2_line,handle_e1_recon, handle_e2_recon, handle_e1_proj,  handle_e2_proj
]
labels = [
    "e1", "e2", "data", "e1's line",  "e2's line",
    "e1 reconstruction","e2 reconstruction", "e1 1D projection",
     "e2 1D projection"
]


# 5. Create the legend below the plot
# We use fig.legend() instead of ax.legend() to place it relative to the whole figure.
# 'lower center' places it at the bottom, and bbox_to_anchor adjusts its vertical position.
# 'ncol' sets the number of columns for a more compact layout.
# handler_map is used to handle the tuple of artists for the arrow.
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, frameon=False, handler_map={tuple: HandlerTuple(ndivide=None)})

# 6. Adjust the plot layout to make space for the legend
# We call tight_layout first, then adjust the bottom margin.
plt.tight_layout()
plt.subplots_adjust(bottom=0.25) # Increase bottom margin to prevent legend overlap


plt.show()
```
