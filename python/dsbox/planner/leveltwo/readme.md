=======
How to run L2 Planner

python run.py -p [path to problem directory] -l library
example python run.py -p ../data/o_38 -l library

Task type: classification
Data profile: Missing Values, Not Numerical

L1 assumed Pipelines:
-------------
[[Logistic Regression], [SVM], [KNNClassifier], [Gaussian Naive Bayes], [Bernoulli Naive Bayes], [Multinomial Naive Bayes]]
-------------
Logistic Regression requirement: Numerical
-> Adding Label Encoder
SVM requirement: Numerical
-> Adding Label Encoder
SVM requirement: No Missing Values
-> Adding Imputer
KNNClassifier requirement: Numerical
-> Adding Label Encoder
Gaussian Naive Bayes requirement: Numerical
-> Adding Label Encoder
Bernoulli Naive Bayes requirement: Numerical
-> Adding Label Encoder
Multinomial Naive Bayes requirement: Numerical
-> Adding Label Encoder
/usr/local/lib/python2.7/site-packages/numpy/lib/arraysetops.py:216: FutureWarning: numpy not_equal will not check object identity in the future. The comparison did not return the same result as suggested by the identity (`is`)) and will change.
  flag = np.concatenate(([True], aux[1:] != aux[:-1]))

L2 Pipelines:
-------------
[([Label Encoder, Logistic Regression], 0.92549668874172186), ([Imputer, Label Encoder, SVM], 0.92384105960264906), ([Label Encoder, KNNClassifier], 0.92384105960264906), ([Label Encoder, Gaussian Naive Bayes], 0.44205298013245031), ([Label Encoder, Bernoulli Naive Bayes], 0.91887417218543044), ([Label Encoder, Multinomial Naive Bayes], 0.51821192052980136)]
You can play with models.json and glue.json in the library directory to add/modify primitives