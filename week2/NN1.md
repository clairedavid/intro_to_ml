# 6. Neural Networks Part I
The present and following lectures introduce the famous Artificial Neural Networks, a computing system inspired by our connected neurons in the organic grey matter that makes animal brains.  

Why are they so popular?  

Because Neural Networks have great qualities. They are:
* __versatile__: they deliver for various ranges of problems and modeling situations
* __powerful__: they show amazing performance
* __scalable__: they can work with very large datasets, some having more than a hundred billions parameters!

Artificial Neural Networks (ANN) are everywhere in modern day-to-day life. They serve to classify billions of images (Google Image), or to recommend products or videos, correct the grammar in your favourite messenger application, recognize speech and outperform the best Go or Chess players (AlphaGo and AlphaZero developed by artificial intelligence research company DeepMind). 

In sciences, neural networks are used across many disciplines for various situations. To quote only a few: medical diagnosis, climatic forecasting, natural disaster management. The list is really long. Neural Networks are also very good mathematicians when it comes to solve complex equations. They are a great tool mastering what physicists often fail: [solving complex partial differential equations](https://www.quantamagazine.org/latest-neural-nets-solve-worlds-hardest-equations-faster-than-ever-before-20210419). Those equations are notoriously difficult and require millions of CPU hours. Example: how to model a fluid flowing around an object. Neural Network would deliver within seconds when other traditional solvers would take dozens of hours of computation. 

In experimental particle physics, neural networks can be met at different stages of a data analysis. They are mostly used for pattern recognition tasks, such as identifying the particles present in a collision. Elementary particles called quarks, constituents of protons and neutrons inside the atom's nucleus, come in different types (up, down, charm, strange, top, bottom) and cannot be seen directly; they form a spray of other particles that are picked by the detector's electronics as a 'jet' (due to its conical shape). The Deep Learning algorithm DL1 is capable, by studying the features of the jets, to identify if the originating quark producing them are of type bottom (it is referred to b-tagging). Behind the scene is a Neural Network fed by all jet properties as input features.

Neural networks start from the simplest unit, the perceptron, and span various architectures baring names such as Convolutional Neural Networks (CNN), Generative Adversarial Neural Network (GAN), Graph Neural Network (GNN)... It is a very large family in Machine Learning, growing constantlty with new members created by brilliant Machine Learning experts.

In this first lecture, we will cover the basics:

* __What motivates neural networks?__
* __How are neural networks represented as a model?__  
* __What are the activation functions?__  
* __How to feed a neural network to get output values? (aka Forward Propagation)__

Let's go!




