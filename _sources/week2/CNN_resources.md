# Learn CNNs 
Self-serve learning at your own pace with great videos.

```{admonition} Important
:class: warning 
Before the guest lecture, please watch at least:  
- One of the videos in the Section "Essentials"  
- One of the videos in the Section "How it works"

After watching the videos, you should be able to answers the questions below. Read them carefully now as they shall provide a guide on what to pay attention to.
```

## Guiding Questions

### General questions
* What are the motivations of CNN?
* Why are they ideal for image processing? (3 points)
* What is a convolutional layer?
* What are the fundamental components of a convolutional neural network (CNN) in terms of its architecture?

### Detailed questions
* What is a filter? 
* What is a kernel?
* How are weights represented in a CNN?
* What is a feature map?
* What does pooling do?
* What does padding do?


## Warm-up (optional)
Refresh your knowledge about convolution with the wonderful animations from the math YouTube channel 3Blue1Brown.
The part relevant for this course is from the start until 14 minutes 40.  

__<center><a href="https://www.youtube.com/watch?v=KuXjwB4LzSA" target="_blank">But what is a convolution?</a></center>__

```{image} ../images/lec09_2_convo_3blue1brown.png
:width: 400px
:align: center
```  



## Essentials

### Working through a small example with StatQuest (preferred)
In this video, Josh Starmer from StatQuest walks through a simple example of CNN classifying between a O or X image symbol. The main ideas behind filters and pooling are highlighted. You will see that Josh likes to sing and says "bam" a lot.  

<center>  

__<a href="https://www.youtube.com/watch?v=HGwBXDKFk9I" target="_blank">Video: Image Classification with Convolutional Neural Networks, StatQuest</a>__  

<a href="https://www.youtube.com/watch?v=HGwBXDKFk9I" target="_blank"><img src="https://img.youtube.com/vi/HGwBXDKFk9I/0.jpg" alt="StatQuestCNN" class="bg-primary mb-1" width="400px"></a>
</center>
&nbsp;  

### Feature extractions, Deeplizard
This video shows the roles of filters for feature extraction. It also shows an example of a convolution implemented in an spreadsheet.

<center>  

__<a href="https://www.youtube.com/watch?v=YRhxdVk_sIs" target="_blank">Video: Convolutional Neural Networks explained, Deeplizard</a>__  

<a href="https://www.youtube.com/watch?v=YRhxdVk_sIs" target="_blank"><img src="https://img.youtube.com/vi/YRhxdVk_sIs/0.jpg" alt="DeepLizard" class="bg-primary mb-1" width="400px"></a>
</center>  


## How CNNs work

### Stanford Lecture 5 | Convolutional Neural Networks
Stanford University School of Engineering offers a series of lectures. In Lecture 5, Serena Yeung put __emphasis on the layer dimensions__  before and after each operation. Pause the video and try to answer first before the answer is revealed!
````{margin}
```{note}
The depth is mentioned in the video but not defined. The depth of an image refers to the number of channels it has, with a grayscale image having a depth of 1 and a color image having a depth of 3 in the [RGB color model](https://en.wikipedia.org/wiki/RGB_color_model). In a neural network, the depth of a layer refers to the number of channels in a feature map, with each channel representing a different feature.
```
````
From start to minute 14:00 $\rightarrow$ Review on history and applications  
From minute 14:00 onward $\rightarrow$ How CNNs work

<center>  

__<a href="https://www.youtube.com/watch?v=bNb2fEVKeEo" target="_blank">Video: Lecture 5 | Convolutional Neural Networks, Stanford University</a>__  

<a href="https://www.youtube.com/watch?v=bNb2fEVKeEo" target="_blank"><img src="https://img.youtube.com/vi/bNb2fEVKeEo/0.jpg" alt="StanfordLecture5" class="bg-primary mb-1" width="400px"></a>
</center>
&nbsp;  

### The Math of Intelligence | CNN by Siraj Raval
This video presents the big blocks of CNNs, then the role of each layer.  

__What I like__  
Siraj defines most of the jargon: depth, pooling, dropout. Big bonus: from minute 32 there is a guided visit of a CNN coded in python from scratch, without any of libraries such as Keras, PyTorch or TensorFlow.  

__What I dislike__  
The use of the workd "magic" ... There is no magic, it's all about math!

<center>  

__<a href="https://www.youtube.com/watch?v=FTr3n7uBIuE" target="_blank">Video: Convolutional Neural Networks - The Math of Intelligence, Siraj Raval</a>__  

<a href="https://www.youtube.com/watch?v=FTr3n7uBIuE" target="_blank"><img src="https://img.youtube.com/vi/FTr3n7uBIuE/0.jpg" alt="SirajRavalCNN" class="bg-primary mb-1" width="400px"></a>
</center>  
&nbsp;  


## If you are curious (optional)

More links below for those eager to convolve themselves with the joys of CNNs.

### The AlexNet breakthrough
AlexNet is a deep convolutional neural network (CNN) introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton in their paper "ImageNet Classification with Deep Convolutional Neural Networks" in 2012. The network achieved a top-5 test error rate of 15.3% on the ImageNet dataset, which was a significant improvement over the previous state-of-the-art at the time. The success of AlexNet marked the beginning of the deep learning revolution in the field of computer vision.

Learn more about the architecture and features of AlexNet on the website www.paperwithcode.com, where you can also access an implementation of the network with PyTorch. The code is surprisingly short; 216 lines!
<center> 

__<a href="https://paperswithcode.com/method/alexnet" target="_blank">Paper with code: AlexNet</a>__
</center>  
&nbsp;  

### Stanford Lecture 9 | CNN Architectures
In Stanford Lecture 9, Serena Yeung reviews the architectures of the ImageNet winners: AlexNet, VGG, GoogLeNet, and ResNet. It is an advanced topic and you may need to pause the video and do additional research to understand the technical terms used. It is recommended to first read the AlexNet paper linked above to familiarize yourself with the terminology, concepts and architecture illustrations. After watching this video, you will have a comprehensive knowledge of the various CNN architectures and their successes in the field of computer vision.

<center>  

__<a href="https://www.youtube.com/watch?v=DAOcjicFr1Y" target="_blank">Video: Lecture 9 | CNN Architectures, Stanford University</a>__  

<a href="https://www.youtube.com/watch?v=DAOcjicFr1Y" target="_blank"><img src="https://img.youtube.com/vi/DAOcjicFr1Y/0.jpg" alt="SirajRavalCNN" class="bg-primary mb-1" width="400px"></a>
</center>  
&nbsp;  

### CNN with Keras and TensorFlow - Live code! By sentdex
If you want to see a CNN implemented live on a Jupyter Notebook in front of your eyes: this is the video! The code starts at minute 6:00 (before sentdex gives a brief review). 

<center>  

__<a href="https://www.youtube.com/watch?v=WvoLTXIjBYU" target="_blank">Video: CNN - Deep Learning basics with Python, TensorFlow and Keras, sentdex</a>__  

<a href="https://www.youtube.com/watch?v=WvoLTXIjBYU" target="_blank"><img src="https://img.youtube.com/vi/WvoLTXIjBYU/0.jpg" alt="SirajRavalCNN" class="bg-primary mb-1" width="400px"></a>
</center>  
&nbsp;  

```{note}
A live Q&A session with Dr. Nachman will be organized following his lecture, likely in the evening, due to the time difference. You will be notified. 
```