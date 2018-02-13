# Bootcamp

Bootcamp is meant to provide a ML/DL/NLP context as fast as possible given some basic requisites.

## Requirements

Before diving into learning it is good that you refresh your knowledge of:

* Python 3.x.: [Python Tutorial](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/00.%20Python%20Tutorial.ipynb), [A collection of not-so-obvious Python stuff](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/00.%20Python%20Tutorial.ipynb)
* Linear algebra: [Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra)
* Probality and statistics: [Book](http://heather.cs.ucdavis.edu/~matloff/132/PLN/probstatbook/ProbStatBook.pdf)

## Online tutorials and courses

Start by shorter tutorials to get an overall understanding of the area. I recommend you to go over these tutorials without getting into details.

* [Udacity's Deep Learning by Google](https://udacity.com/course/deep-learning--ud730)
> Basic and fast paced but you can take it in one/two days and get an overall feeling of DL.

* [Kaggle's Machine Learning tutorial for beginners](https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners).

Now you are ready for a more challenging one:

* [Practical Deep Learning For Coders, Part 1](http://course.fast.ai/index.html)

Once you master the essential elements you can move into deeper waters to also grasp the NLP elements we need. So far, I recommend you to take one of these:

* [Oxford Deep NLP 2017 course](https://github.com/oxford-cs-deepnlp-2017/lectures)
* [Stanford's CS224N *Natural Language Processing with Deep Learning* videos]( https://www.youtube.com/playlist?list=PLqdrfNEc5QnuV9RwUAhoJcoQvu4Q46Lja)

> So far, I prefer Oxford's but I am still deciding.

#### Other tutorials to check

* [Machine Learning with Python](https://www.youtube.com/watch?v=OGxgnH8y2NM&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

## Books

In practice, it is unlikely that you have the time to read whole books. I am listing some here that I personally like and you could use as reference of support when taking the tutorials.

### If you are in a "hurry".

1. Nikhil Buduma (2017) Fundamentals of Deep Learning. O'Reilly Media, Sebastopol, CA.
> Recommended if you want a fast introduction, albeit losing some (mostly theoretical) details. This book is perhaps too succinct. On the other hand, if you are a programmer, I think you want a direct explanation and running code with of the whole picture and then drill down into the details.

### Books with online materials

‎* S. Raschka and V. Mirjalili (2017) Python Machine Learning, 2nd Edition [GitHub repo](https://github.com/rasbt/python-machine-learning-book-2nd-edition)

### Reference books

1. Hastie, Tibshirani and Friedman (2009) The Elements of Statistical Learning (2nd edition) Springer-Verlag. [web](http://web.stanford.edu/~hastie/ElemStatLearn/)
> A solid book to the foundations of machine learning.
2. Grégoire Montavon, Geneviève B. Orr and Klaus-Robert Müller (2012) Neural Networks: Tricks of the Trade (Second edition). Springer LNCS 7700.
> Different applications of neural networks. I liked the book when I read it.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT press. [online in html-ish form](http://www.deeplearningbook.org/) and as [pdf](https://github.com/janishar/mit-deep-learning-book-pdf)
> Perhaps it is too dense for a begginer. It includes algebra and probabilistic "refreshers" at the beginning.
4. Dan Jurafsky and James H. Martin (2017) *Speech and Language Processing* [3rd edition in progress available](https://web.stanford.edu/~jurafsky/slp3/)
> A classic NLP book, the draft of the 3rd edition is online with some chapters missing.
2. Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze, *Introduction to Information Retrieval*, Cambridge University Press. 2008. [online](https://nlp.stanford.edu/IR-book/)
> Classical book with all the basics.

## Libraries to master at this level

* Numpy/scipy
* Matplotlib/seaborn
* scikit-learn
* pandas
* Tensorflow and Keras
* Pytorch
* Jupyter/IPython notebooks
* [Natural Language Toolkit](http://www.nltk.org)

See [Technical setup](Technical.md) for details on installing.

## Bootcamp skill checklist

* **Machine Learning basics**
  * Linear regression.
  * Linear classification and logistic regression.
  * Experimental methodology: train/test/validate, cross-validation.
  * Performance assessment: Machine Learning and Performance Evaluation — Overcoming the Selection Bias
  * Reporting results and visualization.
  * Parameter optimization, grid search.

* **Multilayer Perceptrons (MLPs)** ([slides](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/04.%20Artificial%20neural%20networks.ipynb)):
  * Need for more than one layer of neurons.
  * Gradient descent and error backpropagation.
  * Stochastic gradient descent.
  * Designing neural networks, choice of activation functions on each layer.

* **Deep Learning** ([slides](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/06.%20Deep%20Learning.ipynb)):
  - Why can't you train a *plain* MLP with many layers: vanishing gradients.
  - Going deep one layer at a time: stacked auto-encoders.

* **Convolutional Neural Networks (CNNs)**:
  - Why MLPs can't handle images?
  - Notion of weight sharing.
  - Convolutional layer,
  - Pooling layer.

* **Recurrent Neural Networks (RNNs)**
  - Basic from MLPs to RNNs.
  - RNN challenges.
  - Long short-term memories (LSTMs).

* **Natural Language Processing**
  - Statistical/Bayesian concepts.
  - Understanding HMMs.
  - NLP problems: part-of-speech tagging, named-entities recognition, etc.
  - N-gram concepts.
  - Bag of words.
  - TF-IDF.
  - word2vec, [resources](https://github.com/clulab/nlp-reading-group/wiki/Word2Vec-Resources)
  - glovec, doc2vec.
  - In progress…
