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

# AutoEncoders

AutoEncoders are neural networks that *learn* to perform dimensionality reduction for a given dataset. They consist of an encoder $\mathrm{Enc}_{\phi}$ network which maps the data to a lower dimensional representation and a decoder network $\mathrm{Dec}_{\theta}$ which reconstructs data from these lower dimensional samples. These networks are simultanousely optimised to minimise a reconstruction loss:

```{margin}
AutoEncoders (and especially their more advanced variants) can be used for far more than just dimensionality reduction but it is what we'll be focusing on.
```


```{math}
\mathcal{L}(\phi,\theta) = \frac{1}{N}\sum_{n=1}^{N}{\left\Vert x_n - \mathrm{Dec}_{\theta}(\mathrm{Enc}_{\phi}(x_n)) \right\Vert}_{2}^2
```

```{warning}
In this section it is assumed that you're already comfortable with the basics of neural networks. If that's not the case it is highly recommended that you review the fundamentals of neural networks before proceeding.
```

In order to understand how AutoEncoders are trained we'll go through an example with the Handwritten Digits dataset from Scikit-Learn. This dataset is similar to MNIST but the images are only $8\times 8$.

```{code-cell}
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

fig, axarr = plt.subplots(1,3)

axarr[0].imshow(digits.images[64],cmap="gray")
axarr[0].set_title(digits.target[64])
axarr[0].set_xticks([])
axarr[0].set_yticks([])

axarr[1].imshow(digits.images[1337],cmap="gray")
axarr[1].set_title(digits.target[1337])
axarr[1].set_xticks([])
axarr[1].set_yticks([])


axarr[2].imshow(digits.images[5],cmap="gray")
axarr[2].set_title(digits.target[5])
axarr[2].set_xticks([])
axarr[2].set_yticks([])
plt.show()
```

Since the dataset is tiny, we can load all of it into device memory:
```{code-cell}
import torch
torch.manual_seed(34)

#Use a GPU if available
if torch.cuda.is_available():
    torch.cuda.manual_seed(33)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
flat_imgs = torch.as_tensor(digits.data, device=device, dtype=torch.float32)
```


Next we'll write our neural network.

```{code-cell}
from typing import List
import torch.nn as nn
import numpy as np

class MLPBlock(nn.Module):
    """
    Generic MLP block. 
    
    Consists of a series of linear layers followed by ReLU activation (with the exception of the last layer).

    Args:
        features (List[int]): A list of the feature dimensions throughout the block.

    """
    def __init__(self, features: List[int]):
        super().__init__()
        layers=[]
        for ldx, (f_in, f_out) in enumerate(zip(features[:-1], features[1:])):
            layers.append(nn.Linear(f_in, f_out))
            # No relu on last layer's output
            if ldx != len(features)-2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(
            *layers
        )
    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, features: List[int]):
        super().__init__()
        self.encoder = MLPBlock(features)
        self.decoder = nn.Sequential(MLPBlock(list(reversed(features))), nn.ReLU())
    
    def encode(self, x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x)) 
    
```

Then we'll initialise the model, optimiser and loss function.

```{code-cell}
autoencoder = AutoEncoder([64, 128, 128, 2]).to(device)
optimiser = torch.optim.Adam(autoencoder.parameters(),1e-3)
loss_fn = nn.MSELoss()
```

Finally, we'll write a basic training loop.

```{code-cell}
rng_gen = np.random.default_rng(432)
n_epochs = 50
batch_size = 64
indices = np.arange(flat_imgs.shape[0])


for epoch in range(n_epochs):
    if epoch%10==0:
        losses = []
    rng_gen.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        bdx = indices[i:i+batch_size]
        batch = flat_imgs[bdx]

        optimiser.zero_grad()
        output = autoencoder(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())  
    if epoch%10==0 or epoch==n_epochs-1: 
        print(f'[Epoch {epoch}] Loss: {np.mean(losses):.3f}')
```


Let's check our autoencoder's ability to reconstruct the images!

```{code-cell}
autoencoder.eval()
test_idxs = rng_gen.integers(0,len(indices),3)
with torch.no_grad():
    reconstructed = autoencoder(flat_imgs[test_idxs]).cpu()
originals = digits.images[test_idxs]
fig, axarr = plt.subplots(1, 6,figsize=(12,2))

for i in range(3):
    axarr[2*i].set_title('Reconstructed')
    axarr[2*i].set_xticks([])
    axarr[2*i].set_yticks([])
    axarr[2*i].imshow(reconstructed[i,:].reshape(8,8),cmap='gray')
    
    axarr[2*i+1].set_title('Original')
    axarr[2*i+1].set_xticks([])
    axarr[2*i+1].set_yticks([])
    axarr[2*i+1].imshow(originals[i,:],cmap='gray')

plt.show()
```

That's pretty good considering that our latent dimension is only 2!

Next we can visualise our latent space by encoding the entire dataset.

```{code-cell}
autoencoder.eval()
with torch.no_grad():
    encoded_data = autoencoder.encode(flat_imgs).cpu()

fig,ax=plt.subplots()
scatter=ax.scatter(encoded_data[:,0], encoded_data[:,1], c=digits.target, cmap='tab10')
fig.colorbar(scatter, ax=ax,ticks=range(10), label='Digit')
ax.grid(True)
ax.set_xlabel(r'$z_1$')
ax.set_ylabel(r'$z_2$')
plt.show()


```

Whilst we are able to reconstruct the data quite well, the latent representation doesn't nicely separate the different digits. This is a consequence of the flexibility of AutoEncoders, since they are much more unconstrained there are many more functions that they can learn which achieve the same reconstruction loss. However, these different functions may have vastly different latent representations of the data. Therefore, if we desire a particular structure in our latent space, that structure is usually encouraged by adapting the loss function, introducing constraints or modifying the neural network architecture. 

```{admonition} Exercise
:class: seealso

Try running this code with different random seeds. <br/>
1.) How does this affect the latent space? <br/>
2.) Do models with low reconstruction loss have similar latent spaces?
```


Hopefully, this has given you a basic idea of how AutoEncoders work. In order to inspire your further, I'd like to give a brief overview of some more sophisticated autoencoders and their applications.

**Variational AutoEncoders** treat the latent projection of each input as a (typically Gaussian) random variable and incorporate an additional term in the loss function which ensures that these latent variables are distributed similarly to a standard, easy to sample from distribution. Their applications include generative modelling and outlier detection. See {cite:p}`prince2023understanding` for a first introduction and {cite:p}`Kingma_2019` for an in-depth review.

**Sparse AutoEncoders** are constrained so that the activations in their hidden layers are sparse. This means that for any given input, only a small number of neurons are allowed to have non-zero output. Sparse AutoEncoders have become an important tool in LLM interpretability research. Typically, single-layer Sparse AutoEncoders are trained on the activations of an LLM's layer. After training, researchers analyse the patterns of sparse activation in the *AutoEncoder's* hidden layer as input is passed through the LLM. See this recent ICLR paper {cite:p}`huben2024sparse` and this [blogpost from Anthropic](https://transformer-circuits.pub/2023/monosemantic-features/) for more.


**Denoising AutoEncoders** add noise to their input before passing it through the encoder and aim to reconstruct the uncorrupted input. This additional regularisation is thought to lead to more robust and useful latent representations. Additionally, the residuals of Denoising AutoEncoders trained with Gaussian corruption and MSE loss approximate the score function of the data $\nabla_{x}\log p(x)$. This score function can in turn be used to generate new samples. They are closely linked to *Diffusion Models*, which are the state-of-the-art generative models for image and video. Diffusion Models learn the score-function for the data at multiple levels of pertubation and sample by iteratively guiding noise samples toward the data distribution with these different score functions. See {cite:p}`pml1Book` for a first introduction and [Yang Song's blogpost](https://yang-song.net/blog/2021/score/) for an in-depth explanation of the connections to Diffusion Models.

```{bibliography}
```