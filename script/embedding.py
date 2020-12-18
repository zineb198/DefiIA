import pandas as pd
from wordembedding import WordEmbedding
import os


# Define PATH files
DATA_PATH = '/Users/cecile/Documents/INSA/5A/DefiIA/data/'  #TODO

DATA_CLEANED_PATH = DATA_PATH+'/cleaned/'
if not os.path.exists(DATA_CLEANED_PATH):
    print('You have to run cleaning.py before this !')

DATA_MODELS_PATH = DATA_PATH+'/models/'
if not os.path.exists(DATA_MODELS_PATH):
    os.makedirs(DATA_MODELS_PATH)
    os.makedirs(DATA_MODELS_PATH+'/word2vec')
    os.makedirs(DATA_MODELS_PATH + '/fasttext')


# Reading files
train_df = pd.read_csv(DATA_CLEANED_PATH+'cleaned_train.csv', index_col=0)
test_df = pd.read_csv(DATA_CLEANED_PATH+'cleaned_test.csv', index_col=0)
train_label = pd.read_csv(DATA_CLEANED_PATH+'train_label.csv', index_col=0)
array_token = [line.split(" ") for line in train_df["description_cleaned"].values]


# Define params models
params_word2vec = dict(sentences=array_token, iter=1, sg=0, size=5)

# Training & save model
we = WordEmbedding('word2vec', params_word2vec, DATA_MODELS_PATH)
we.train_save()

