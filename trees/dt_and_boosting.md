# Decision Trees

This lecture is about decision trees, root nodes, leaves and forests. No, we haven’t switched to a course on dendrology or xylology. We’re still in machine learning, just borrowing vocabulary from a (convenient) vegetation analogy.

Decision trees are powerful algorithms. But also limited, as overfitting datasets by design. We will see how decision trees work and what restrictions they need.

We will also see how a collection of simple decision trees, aka a random forest, can actually generate better predictions than each learner. Such a technique is called ensemble learning, popularily coined as the wisdom of crowds (trees can be wise too).

Growing a forest is good, boosting it is better. The boosting exploits ensemble learning by sequentially correcting the previous predictions of simple learners at the next iterations, the decision trees receiving information on where predictions failed. We will cover two main boosting algorithms, AdaBoost and XGBoost, counting among the more popular methods in machine learning today.