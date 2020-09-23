import pandas
import gensim
import gensim.downloader as api
import numpy
from sklearn.model_selection import train_test_split
import keras.preprocessing.sequence as sequence
from keras import Sequential
from keras.layers import LSTM, Dense
from nltk.tokenize import word_tokenize
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import nltk
nltk.download('punkt')
import gc

SIZE = 100000

def tokenize_review(review):
    """Function to tokenize each review"""
    review = review.lower()  # convert to lowercase
    tokens = word_tokenize(review)  # use punkt to tokenize review
    # tokens = [x for x in tokens if x not in string.punctuation] # step to remove punctuation
    # tokens = [x for x in tokens if x not in stop_words] # step to remove stopwords
    return tokens

def postprocess(data, n=1000000):
    """Function to process reviews for Gensim W2V."""
    #data = data.head(n)
    data['tokens'] = data['text'].progress_map(tokenize_review)
    return data

"""
DATA PREPARATION CODE
"""
df = None
for dataframe in pandas.read_json("yelp_academic_dataset_review.json", lines=True, chunksize=SIZE, nrows=SIZE):
  df = dataframe
  break
df.drop(['review_id', 'user_id', 'business_id', 'useful', 'funny',
       'cool', 'date'], axis=1, inplace=True)     # drop useless data from frame to save space, time
df = df[df.stars.isnull() == False]
df['stars'] = df['stars'].map(int)
df = df[df.text.isnull() == False]
print('dataset loaded with shape:', df.shape)

df = postprocess(df)

# Initializing pre-trained Word2Vec embedding model
#w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#corpus = api.load('text8')
#w2v_model = gensim.models.Word2Vec(corpus)
w2v_model = api.load("word2vec-google-news-300")

# Generating embeddings for all the reviews in the dataset

data = list()
# For each review in the corpus
for row in range(len(df)):
    review = []
    # For each word in the review
    for t in df['tokens'][row]:
        # Append the w2v vector for the word to the review embedding
        try:
            #review.append(w2v_model.get_vector(t.lower()))
            review.append(w2v_model[t.lower()])
        except KeyError:
            continue
    data.append(review)
data = numpy.array(data)
labels = numpy.array(df['stars'])

print(len(data))
print(len(labels))
print(len(data[0][0]))

vocab_size=3000000
embedding_dim=300

# Neural network classifier model

model = Sequential()
model.add(LSTM(100, input_dim=embedding_dim))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Shuffle the data and labels *in the same order* to prevent overfitting
seed = numpy.random.get_state()
numpy.random.shuffle(data)
numpy.random.set_state(seed)
numpy.random.shuffle(labels)

"""
max_input_words = 0
# Find the longest review to pad the others to match its length
for row in range(len(df)):
    if len(df['tokens'][row]) > max_input_words:
        max_input_words = len(df['tokens'][row])

print(max_input_words)

if max_input_words == 0:
    max_input_words = 300
"""

max_input_words = 300
traind, vald, trainl, vall = train_test_split(data, labels)
del data, labels, w2v_model, df
print(gc.get_count())
gc.collect()
print(gc.get_count())
traind = sequence.pad_sequences(traind, maxlen=max_input_words)
vald = sequence.pad_sequences(vald, maxlen=max_input_words)

#train_gen = ((numpy.array(sequence.pad_sequences(traind[x], maxlen=max_input_words)), numpy.array(trainl[x])) for x in range(len(trainl)))
#val_gen = ((numpy.array(sequence.pad_sequences(vald[x], maxlen=max_input_words)), numpy.array(vall[x])) for x in range(len(vall)))

"""
traind = numpy.array(traind)
vald = numpy.array(vald)
trainl = numpy.array(trainl)
traind = numpy.array(vall)
"""
"""
def train_gen(traind, trainl):
    for x in range(len(trainl)):
        traind[x] = numpy.array(sequences.pad_sequences(traind[x], maxlen=max_input_words))
        trainl[x] = numpy.array(trainl[x])
        yield (traind[x], trainl[x])

def val_gen(vald, vall):
    for x in range(len(vall)):
        vald[x] = numpy.array(sequences.pad_sequences(vald[x], maxlen=max_input_words))
        vall[x] = numpy.array(vall[x])
        yield (vald[x], vall[x])
"""

model.fit(traind, trainl, verbose=1)
#model.fit_generator(train_gen)

del traind, trainl
print(gc.get_count())
gc.collect()
print(gc.get_count())

model.evaluate(vald, vall, verbose=1)
#model.evaluate_generator(val_gen)
