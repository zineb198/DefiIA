import pandas as pd
from wordembedding import WordEmbedding
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


# Define PATH files
DATA_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/'  # TODO
DATA_MODELS_PATH = '/Users/cecile/Documents/INSA/DefiIA/models/'  # TODO
DATA_CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned')
DATA_PATH_TFIDF = os.path.join(DATA_MODELS_PATH, 'tf-idf')

if not os.path.exists(DATA_CLEANED_PATH):
    print('Try again ! You have to run cleaning.py before this !')
if not os.path.exists(DATA_MODELS_PATH):
    os.makedirs(DATA_MODELS_PATH)
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'tf-idf'))
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'word2vec'))
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'fasttext'))


# Reading files
params_cl = '_stem'  # TODO : '_stem' if stemming=True or '_lem' if lemmatizer in cleaning

train_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'train_cleaned' + params_cl + '.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'test_cleaned' + params_cl + '.csv'), index_col=0)

train_label = pd.read_csv(os.path.join(DATA_PATH, 'train_label.csv'), index_col=0)

# TF-IDF on cleaned text
transformer = TfidfVectorizer().fit(train_df["description_cleaned"].values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(train_df["description_cleaned"].values)
X_test = transformer.transform(test_df["description_cleaned"].values)

# Saving results
pickle.dump(X_train, open(os.path.join(DATA_PATH_TFIDF, 'X_train.pickle'), 'wb'))
pickle.dump(X_test, open(os.path.join(DATA_PATH_TFIDF, 'X_test.pickle'), 'wb'))
