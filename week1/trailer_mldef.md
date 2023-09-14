# What is Machine Learning?

## Foreword
Before heading to the very formal definition, let's mention important points.

First, __Machine Learning is a technique__. It is a method that is designed to solve certain types of problems. Not all problems (more in the next section). 
Machine learning is a sub-field of artificial intelligence (AI), an academic discipline founded in 1956. In the recent years the field has been booming, with more and more applications and dedicated research flourishing in many areas, both in academia and in industry. Machine learning has become a buzz word and for some enthusiastic programmers, a cool toy to play with. There is even [art](https://ml4a.net/) created using machine learning algorithms. But we should keep in mind that it is, before all, a technique. And like all techniques and tools, it poses ethical questions on when and how to use it (we will come back to ethical questions at the end of this course).

Second: in its classical sense, __a computer program is dumb__. It is a set of instructions written by a human. The program does nothing more than executing these instructions. The advantage we have compared to human brains doing the same is robustness and speed. If the human codes wrong instructions, the program is not smart enough to know the wrongness of rightness of things according to our standards. A machine learning algorithm is different, yet care should be done before considering it smart. The intelligence is rather in the (human) brains who invented the algorithms in the first place. Even with the best machine learning program at hand, if the input data is corrupted or not adequate, one cannot expect proper outcome or worse, misinterpret the answer. Therefore it is essential to understand what is behind the 'black box' of a machine learning application (hint: a lot of math), what are the goals and anticipate the possible output. It is a crucial step before developing more complex algorithms. This is what we will cover in the course.

```{admonition} Exercise
:class: dropdown
On your own, write your definition of intelligence. What would be the properties of an intelligent system?  
Then share it to your neighbours to compare their definition and list of properties with yours.
```

## Formal definitions

A first - and old - definition comes from Arthur Samuel, an American pioneer in the field of computer gaming.

__Machine Learning: field of study that gives computers the ability to learn without being explicitly programmed.__  

This is somewhat redundant, as the definition uses the verb to 'learn' in it. What is learning? And how to quantify it? 

Let's see what Tom Mitchell has to say. He is an American computer scientist and defined machine learning in the following statement in 1998:

````{prf:definition}
:label: mldef
A computer program is said to learn from __experience E__ with respect to some __task T__ and some __performance measure P__, if its performance on __T__, as measured by __P__, improves with experience __E__.
````

Let's illustrate this with Arthur Samuel's original investigation. Back in the days, he wrote a checker - also called draught - playing program.
Let's start with the task: it is here playing a checker game.
The experience E is the repeated action of the program to play against itself for thousands of different games. Through this exposure, the program will eventually see patterns of good or bad tactics from the pieces' positions. The performance P would be the probability of winning the *next game* of checker.
We see here that not only we can see the learning at play (P should increase) but we can quantify it, so we can *compare* the performance of different checker game programs. An important point is that the assessment (win or lose) is done on *new* data. What counts in the experience is novelty, a fresh collection of different game configurations. Feeding the program with the same data duplicated for instance would not help the program to learn. 

Another example at a post station. In some (modern) ones, the envelopes are automatically scanned to sort them by zip code. Here it is a classification problem. The task is the assignment from a picture to a chain of digits and letters. For us humans, we have trained our brain by learning the numbers and the alphabet while young. We perform this recognition without much difficulty, depending on the quality of the handwritting of course. The task here is to assign the correct symbols from a given picture. The experience is to expose the program to envelopes (the input images) and the answer (the correct zip code). The performance is again a probability of the program guessing correctly a new envelope with an unknown zip code. If this probability increases when we expose the program to more scanned envelopes, then we can say the program has learned to read zip codes.

## Machine Learning, statistics and black box
Machine learning uses a lot of statistical concepts at its core.

Statistics is the mathematical study of data: collection, processing, analysis, interpretation and presentation.

How machine learning differs from statistics and statistical modelling? A first aspect is the purpose: machine learning is focused on the result as a technique to make repeatable predictions. It is oriented towards performance, with most of the time a two-step process of _training_ first and then _testing_ the algorithm to know its performance. And this is often happening at the price of interpretability. In statistical modelling, interpretability matters. There is no train nor test steps, rather a meticulous study of all the relationships within the dataset. Stasticians will speak in terms of significance, correlations of variables and confidence intervals, they perform a sort of autopsy of the data. 

In machine learning it is common, sadly, to refer to a program as a so-called black box. "We don't know internally what the machine learning program does, provided it delivers."
This is a dangerous statement. In this course, we will work the maths from the simplest case. This way, when you will build more elaborated algorithms, you will know the inside of each element. Equally important, we will spend time on properly defining the task and performance metrics. Remember, a computer program is dumb. A computer program capable of learning can only learn what it has been programmed to learn. The intelligence, ultimately, is the one of the human programmer.



