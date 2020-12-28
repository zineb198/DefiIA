import pandas as pd
from wordembedding import WordEmbedding
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# Define PATH files
DATA_PATH = '/Users/Morgane/Desktop/5GMM/DefiIA/data/' #'/Users/cecile/Documents/INSA/DefiIA/data/'   #TODO
#DATA_PATH = '/home/cecile/data'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)

DATA_CLEANED_PATH = os.path.join(DATA_PATH, 'cleaned')
if not os.path.exists(DATA_CLEANED_PATH):
    print('Try again ! You have to run cleaning.py before this !')

DATA_MODELS_PATH = '/Users/Morgane/Desktop/5GMM/DefiIA/models/'

#DATA_MODELS_PATH = '/home/cecile/models'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)
if not os.path.exists(DATA_MODELS_PATH):
    os.makedirs(DATA_MODELS_PATH)
    os.makedirs(os.path.join(DATA_MODELS_PATH, 'tf-idf'))

# Reading files
params_cl = '_stem'  # TODO : '_stem' if stemming=True or '_lem' if lemmatizer in cleaning

train_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'train_cleaned' + params_cl + '.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'test_cleaned' + params_cl + '.csv'), index_col=0)

train_label = pd.read_csv(os.path.join(DATA_PATH, 'train_label.csv'), index_col=0)

#on applique tf-idf :
transformer = TfidfVectorizer().fit(train_df["description_cleaned"].values)
print("NB features: %d" %(len(transformer.vocabulary_)))
X_train = transformer.transform(train_df["description_cleaned"].values)
X_test = transformer.transform(test_df["description_cleaned"].values)

#on save les results au bon endroit :
DATA_tf_PATH = DATA_MODELS_PATH + '/tf-idf'
pickle.dump(X_train, open(os.path.join(DATA_tf_PATH,'X_train.pickle'), 'wb'))
pickle.dump(X_test, open(os.path.join(DATA_tf_PATH,'X_test.pickle'), 'wb'))
