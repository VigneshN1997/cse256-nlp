#!/bin/python
# loading required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import spacy
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from tqdm.notebook import tqdm, trange

# downloading necessary word sets
nlp = spacy.load('en')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))

def train_classifier_modified(X, y, c):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=c)
	cls.fit(X, y)
	return cls