# Ensemble Learning

Let's start with a game.

```{admonition} Exercise
:class: seealso
In groups of 4 people, start by guessing individually, then collect your answers on the following questions:  


__Regression__:
<center>What is the instructor's age?</center>

<br>

__Classification__:
<center>⚠️ <small> Skip it if you already know, otherwise the game will be rigged!</small>  

Which fundamental interaction is the strongest:  
Electromagnetism, Weak Force, Strong Force or Gravity?  
</center>

<br>

__A question of your choice__
<center>⚠️ <small> Answer must be checkable online, but no peeking until after the game! </small></center>

<br>  

For the regression, compute the average of your numerical guesses.  
For the classification, count your votes and pick the class collecting most votes.  


Notice anything? Keep your team’s results: we’ll collect them across the class and see what emerges.  
```

We attribute to Aristotle the old adage "the whole is greater than the sum of its parts." The exercise above is an example of the so-called "wisdom of the crowd," where the collective brings a better answer than the individual guesses. Such a technique of getting a prediction from aggregating a collection of predictors is called _Ensemble Learning_. We will see its formal definition in the next section.