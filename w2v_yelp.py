import time
beg = time.time()
print("Imports Begun")

import pandas as pd
# pd.options.mode.chained_assignment = None   # used to prevent throwing a warning
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle
import gensim
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence # ???
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
print("Imports Complete: {} seconds total elapsed".format(round(time.time()-beg), 2))

def ingest_reviews():
    """Fuction to read in the yelp json and convert to pandas df"""
    data = pd.read_json('yelp_academic_dataset_review.json', lines=True) # read json into pd.df
    data.drop(['review_id', 'user_id', 'business_id', 'useful', 'funny',
       'cool', 'date'], axis=1, inplace=True)     # drop useless data from frame to save space, time
    data = data[data.stars.isnull() == False]     # convert null values to False
    data['stars'] = data['stars'].map(int)     # map all star values to int()
    data = data[data.text.isnull() == False]     # convert null values to False
#     data.reset_index(inplace=True)    # create new index
#     data.drop('index', axis=1, inplace=True)     # delete old index
    print('dataset loaded with shape:', data.shape)    # display shape of data for confirmation
    return data


def tokenize_review(review):
    """Function to tokenize each review"""
    review = review.lower()  # convert to lowercase
    tokens = word_tokenize(review)  # use punkt to tokenize review
    # tokens = [x for x in tokens if x not in string.punctuation] # step to remove punctuation
    # tokens = [x for x in tokens if x not in stop_words] # step to remove stopwords
    return tokens


def postprocess(data, n=1000000):
    """Function to process reviews for Gensim W2V."""
    data = data.head(n)
    data['tokens'] = data['text'].progress_map(tokenize_review)
    return data


def label_reviews(reviews, label_type):
    labeled = []
    for i,v in tqdm(enumerate(reviews)):
        label = '%s_%s'%(label_type,i)
        labeled.append(LabeledSentence(v, [label]))
    return labeled

print("Functions Built: {} seconds total elapsed".format(round(time.time()-beg), 2))


# Execution
data = ingest_reviews()
print("Reviews read into Pandas DF: {} seconds total elapsed".format(round(time.time()-beg), 2))
data = postprocess(data)
print("Reviews tokenized for W2V: {} seconds total elapsed".format(round(time.time()-beg), 2))

# Define datasets
x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                    np.array(data.stars), test_size=0.1)
print("Datasets Split: {} seconds total elapsed".format(round(time.time()-beg), 2))

x_Train = label_reviews(x_train, 'TRAIN')
x_Test = label_reviews(x_test, 'TEST')
print("Datasets converted to LabeledSentence: {} seconds total elapsed".format(round(time.time()-beg), 2))

print(x_Train[0])
