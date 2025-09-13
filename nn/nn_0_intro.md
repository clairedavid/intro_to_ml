# Neural Networks: Introduction

The present and following lectures introduce artificial neural networks, computational models inspired by biological neural connections.

Why are they so popular?  

Because neural networks have great qualities. They are:
* __versatile__: they deliver for various ranges of problems and modeling situations
* __powerful__: they show amazing performance
* __scalable__: they can handle very large datasets

From classifying particle collisions in high-energy physics experiments to powering AI assistants with well over a trillion parameters, neural networks are now firmly established, and you probably don’t even realize how often they touch your daily life.

Neural networks start from the simplest unit, the perceptron, and span a wide range of architectures with distinctive names: Convolutional Neural Networks (CNNs) for images, Recurrent Neural Networks (RNNs) for sequences, Generative Adversarial Networks (GANs) for data synthesis, Graph Neural Networks (GNNs) for relational data, Autoencoders and Variational Autoencoders (VAEs) for representation learning, and Transformers for large-scale language and vision tasks. It is a vast and evolving family in machine learning, constantly expanding as new ideas emerge.

In this chapter, we focus on the foundations: the model representation of a neural network and the equations that govern its training. We will walk through the core steps of forward propagation and backpropagation. You will do the math and then implement it by hand during the tutorial. 

<br>

__Learning outcomes:__
* Understand the model representation of a neural network
* Understand activation functions, their properties, and their role
* Recognize common loss and cost functions and their purpose
* Write the equations of feedforward propagation
* Derive the equations of backpropagation
* Understand the constraints imposed by weight initialization
* Compare batch, stochastic, and mini-batch gradient descent
* Grasp the concept of momentum in optimization
* Understand learning rate schedulers
* Be familiar with adaptive optimizers such as RMSProp and Adam

<br>

Let's go!

<br>