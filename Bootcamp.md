# Bootcamp

Bootcamp is meant to provide a machine learning/deep learning/natural language processing context as fast as possible given some basic requisites.

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
* [Deep Learning - The Straight Dope](http://gluon.mxnet.io/index.html)

## Books

In practice, it is unlikely that you have the time to read whole books. I am listing some here that I personally like and you could use as reference of support when taking the tutorials.

### If you are in a "hurry".

1. Yoav Goldberg (2017) *Neural Network Methods for Natural Language Processing*. Synthesis Lectures on Human Language Technologies, Morgan and Claypool Publishers [link](https://doi.org/10.2200/S00762ED1V01Y201703HLT037)

### Books with online materials

* S. Raschka and V. Mirjalili (2017) Python Machine Learning, 2nd Edition [GitHub repo](https://github.com/rasbt/python-machine-learning-book-2nd-edition)

### Reference books

1. Hastie, Tibshirani and Friedman (2009) The Elements of Statistical Learning (2nd edition) Springer-Verlag. [web](http://web.stanford.edu/~hastie/ElemStatLearn/)
> A solid book to the foundations of machine learning.

2. Grégoire Montavon, Geneviève B. Orr and Klaus-Robert Müller (2012) Neural Networks: Tricks of the Trade (Second edition). Springer LNCS 7700.
> Different applications of neural networks. I liked the book when I read it.

3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT press. [online in html-ish form](http://www.deeplearningbook.org/) and as [pdf](https://github.com/janishar/mit-deep-learning-book-pdf)
> It is a little dense for a begginer but very complete. It includes algebra and probabilistic "refreshers" at the beginning.

4. Dan Jurafsky and James H. Martin (2017) *Speech and Language Processing* [3rd edition in progress available](https://web.stanford.edu/~jurafsky/slp3/)
> A classic NLP book, the draft of the 3rd edition is online with some chapters missing.

5. Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze, *Introduction to Information Retrieval*, Cambridge University Press. 2008. [online](https://nlp.stanford.edu/IR-book/)
> Classical CL/IR book with all the basics.

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

## Machine learning skill checklist

* **Machine Learning basics**
  - Linear regression,
  - linear classification and logistic regression,
  - experimental methodology: train/test/validate,
  - cross-validation,
  - performance assessment: [Machine Learning and Performance Evaluation — Overcoming the Selection Bias](https://speakerdeck.com/rasbt/machine-learning-and-performance-evaluation-at-dataphilly-2016), [Video](https://www.youtube.com/watch?v=JlctsNhlNaI),
  - reporting results and visualization, and
  - parameter optimization, grid search, its challenges.

* **Multilayer Perceptrons (MLPs)** ([slides](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/04.%20Artificial%20neural%20networks.ipynb)):
  - Need for more than one layer of neurons,
  - gradient descent and error backpropagation,
  - stochastic gradient descent, and
  - designing neural networks, choice of activation functions on each layer.

* **Deep Learning** ([slides](https://nbviewer.jupyter.org/github/lmarti/machine-learning/blob/master/06.%20Deep%20Learning.ipynb)):
  - Why can't you train a *plain* MLP with many layers: vanishing gradients,
  - Going deep one layer at a time: stacked auto-encoders.

* **Convolutional Neural Networks (CNNs)**:
  - Why MLPs can't handle images?
  - Notion of weight sharing.
  - Convolutional layer.
  - Pooling layer.

* **Recurrent Neural Networks (RNNs)**
  - Basics from MLPs to RNNs
  - RNN challenges
  - Long short-term memories (LSTMs)

## Computational linguistics and natural language processing check list

*Note:* We have taken some comments and links from Steve's [Glossary](CL-NLP-glossary.md).

* **CL/NLP main concepts**
  - information retrieval
  - [NLP overview](https://blog.algorithmia.com/introduction-natural-language-processing-nlp/) [wikipedia](https://en.wikipedia.org/wiki/Natural-language_processing)
  - bag of words
  - [*n*-grams](http://en.wikipedia.org/wiki/Language_model#N-gram_models) See [Bigrams](http://en.wikipedia.org/wiki/#Bigrams), except that n-grams are groups of any number of adjacent characters, words, etc. The large the groups you analyze, the more specific information you can get; however, the corpus size you need goes up even faster.
  - [Term frequency - inverse document frequency (TD-IDF)](http://en.wikipedia.org/wiki/tf-idf).
  - Corpus (dataset) pre-processing.

* **CL/NLP problems**
  - *Sentence segmentation, [part of speech tagging](http://en.wikipedia.org/wiki/Part%20of%20speech%20tagging), and parsing:* Natural language processing can be used to analyze parts of a sentence to better understand the grammatical construction of the sentence.
  - *Deep analytics:* Deep analytics involves the application of advanced data processing techniques in order to extract specific information from large or multi-source data sets. Deep analytics is particularly useful when dealing with precisely targeted or highly complex queries with unstructured and semi-structured data. Deep analytics is often used in the financial sector, the scientific community, the pharmaceutical sector, and biomedical industries. Increasingly, however, deep analysis is also being used by organizations and companies interested in mining data of business value from expansive sets of consumer data.
  - *[Machine Translation](http://en.wikipedia.org/wiki/Machine%20Translation):* Natural language processing is increasingly being used for machine translation programs, in which one human language is automatically translated into another human language.
  - *Named entity extraction:* In data mining, a named entity definition is a phrase or word that clearly identifies one item from a set of other items that have similar attributes. Examples include first and last names, age, geographic locations, addresses, phone numbers, email addresses, company names, etc. Named entity extraction, sometimes also called named entity recognition, makes it easier to mine data. [NER](http://en.wikipedia.org/wiki/Named%20Entity%20Recognition).
  - *Co-reference resolution:* In a chunk of text, co-reference resolution can be used to determine which words are used to refer to the same objects.
  - *Automatic summarization:* Natural language processing can be used to produce a readable summary from a large chunk of text. For example, one might us automatic summarization to produce a short summary of a dense academic article.

> See https://en.wikipedia.org/wiki/Natural-language_processing#Major_evaluations_and_tasks for a more detailed list of problems.
> Descriptions taken from https://www.kdnuggets.com/2015/12/natural-language-processing-101.html

* **Application of statistical/machine learning for NLP**
  - [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
  - [Conditional Random Fields](http://en.wikipedia.org/wiki/Conditional%20Random%20Fields)
  - [Hidden Markov Models (HMM)](http://en.wikipedia.org/wiki/Hidden%20Markov%20Model)
  - [Support Vector Machines](http://en.wikipedia.org/wiki/Support%20Vector%20Machines)
  - Naïve Bayes classifier.

* **Metric embedding**
  - Understanding word embedding - http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/
  - word2vec, [resources](https://github.com/clulab/nlp-reading-group/wiki/Word2Vec-Resources)
  - glovec,
  - doc2vec,
  - pre-trained word embeddings, Facebook embeddings,
  - *add yours here*

* **"Deep" NLP**
  - [NER and the Road to Deep Learning](http://nlp.town/blog/ner-and-the-road-to-deep-learning/)
  - Using CNNs for text classification, common architectures, meaning of 1-D convolution
  - Recurrent neural networks and LSTM for text classification.

* **Character-level representation**
  - Limitations of word-based representation: typos, suffixes, etc.
  -
