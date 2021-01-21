import pandas as pd
from wordembedding import WordEmbedding
import os


# Define PATH files
DATA_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/'  # TODO
DATA_MODELS_PATH = '/Users/cecile/Documents/INSA/DefiIA/models/'  # TODO
DATA_CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned')

if not os.path.exists(DATA_CLEANED_PATH):
    print('Try again ! You have to run cleaning.py before this !')

if not os.path.exists(DATA_MODELS_PATH):
    os.makedirs(DATA_MODELS_PATH)
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'word2vec'))
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'fasttext'))
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'tf-idf'))


# Reading files
params_cl = '_stem'  # TODO : '_stem' if stemming=True or '_lem' if lemmatizer in cleaning

train_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'train_cleaned' + params_cl + '.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'test_cleaned' + params_cl + '.csv'), index_col=0)

train_label = pd.read_csv(os.path.join(DATA_PATH, 'train_label.csv'), index_col=0)

train_array_token = [line.split(" ") for line in train_df["description_cleaned"].values]
test_array_token = [line.split(" ") for line in test_df["description_cleaned"].values]

RNN = False  # TODO : modify the saving format according to the classification RNN or not


# Define params models
for sg in [0, 1]:
    for iter in [10]:
        params_word2vec = dict(sentences=train_array_token, iter=iter, sg=sg, size=300, min_count=1, window=5, hs=0,
                               negative=10)

    # Training & save model
        we = WordEmbedding(word_embedding_type='word2vec', args=params_word2vec,
                           DATA_MODELS_PATH=DATA_MODELS_PATH, array_token=train_array_token,
                           test_array_token=test_array_token)
        we.train_save(RNN)
