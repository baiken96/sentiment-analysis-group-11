#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
beg = time.time()
print("Imports Begun")
# Import Library:
import pandas as pd # data manipulation
# pd.options.mode.chained_assignment = None   # used to prevent throwing a warning
import numpy as np # data manipulation
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
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
import json

# create emptyfolder in path:
#create json files under a specified directory
import dask
import json
import os

path = 'E:\json_data\s'
os.makedirs(path, exist_ok=True)                  # Create data/ directory
b = dask.datasets.make_people()                   # Make records of people
b.map(json.dumps).to_textfiles(path + '*.json')   # Encode as JSON, write to disk


# slice json large file into 8 smaller files

n = 0
chunksize = 1000000
path1 = "D:\slice_data\yelp_academic_dataset_review.json"
path = 'D:\slice_data\s'
for slices in pd.read_json(path1, lines=True,chunksize = chunksize):
    n += 1
    if n <= 7:
        slices.to_json(path + str(n) + '.json')
    else:
        slices.to_json(path + '8.json')
   
print('dataset loaded with shape:', slices.shape, "chunks ", n)    
data = pd.DataFrame(slices)
print('end .........')
data.head(10) 

