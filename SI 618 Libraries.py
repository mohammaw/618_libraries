# Standard Libraries and Utlities

import csv
import re
import string
from collections import Counter, OrderedDict, defaultdict
from io import StringIO
from time import sleep

# Data Processing and Analysis

import numpy as np
import pandas as pd
from pandas import DataFrame

# Statistical Analysis

from scipy import stats, spatial
from scipy.stats import ttest_ind, chi2_contingency, silhouette_samples, silhouette_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.mosaicplot import mosaic
from scipy.cluster.hierarchy import dendrogram

# Machine Learning

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, scale
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer


# Visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from IPython.display import Image

# Natural Language Processing and Web Scraping

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc
from bs4 import BeautifulSoup
import requests
import gensim
import gensim.downloader as api


# Filtering out warnings
import warnings
warnings.filterwarnings("ignore")

# Loading spaCy's English model
nlp = spacy.load('en_core_web_sm')
