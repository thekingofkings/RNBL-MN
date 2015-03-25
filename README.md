# RNBL-MN
Implementation of the Recursive Naive Bayes Learner for sequence classifier


#Keyword

*Weka*, *Recursive Naive Bayes*, *Decision tree*, *Multinomial event model*, *sequence classifier*


## Description
A java class for building and using a recursive Naive Bayes classifier for sequence classification. RNBL-MN is a tree of Naive Bayes classifiers, where each node is a NB classifier based on a multinominal event model.

The RNBL-MN is shown to outperforms C4.5 decision tree learner, and yields accuracies comparable to a SVM using similar information.


##Reference
For more information see,

>Dae-Ki Kang, Adrian Silvescu, Vasant Honavar 
>"RNBL-MN: A Recursive Naive Bayes Learner for Sequences Classification" PAKDD'06.


## Dependencies:
This project relys on the Weka 3.6 NaiveBayesMultinominal classifiers and other assistant functions.

