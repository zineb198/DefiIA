import pandas as pd
import pickle
import numpy as np
from word2vec import Word2Vec
import argparse
import os
import unicodedata
import re
import nltk
import time
import re
import unicodedata
import nltk
from tqdm import tqdm
import gensim


DATA_PATH = "/home/bennis/Bureau/5GMM/AI-Frameworks/DefiIA/data"

parser = argparse.ArgumentParser()
parser.add_argument('--features_dimension', type=int, default=300)
parser.add_argument('--min_count', type=int, default=1)
parser.add_argument('--window', type=int, default=5)
parser.add_argument('--skipgram', type=int, default=1)
parser.add_argument('--hs', type=int, default=0)
parser.add_argument('--negative', type=int, default=10)
args = parser.parse_args()

type = 'CBOW'
if args.skipgram:
    type = 'Skip-Gram'

train_df = pd.read_csv(DATA_PATH+'/cleaned/cleaned_train.csv')
test_df = pd.read_csv(DATA_PATH+'/cleaned/cleaned_test.csv')
train_label = pd.read_csv(DATA_PATH+'/cleaned/train_label.csv')

array_token = [line.split(" ") for line in train_df["description_cleaned"].values]


we = Word2Vec(args = dict(sentences=array_token, sg=args.skipgram, hs=args.hs,
                                  negative = args.negative, min_count=args.min_count,
                                  size=args.features_dimension, window=args.window, iter=10))

print("training..")

model, training_time = we.train()
print("Model ", type, " trained in %.2f minutes"%(training_time/60))

word_vectors = model.wv

DATA_MODELS_PATH = DATA_PATH+'/models/'
if not os.path.exists(DATA_MODELS_PATH):
    os.makedirs(DATA_MODELS_PATH)

#changer les noms pour mettre les paramètres du modèle
#ceci sauve les wordvectors
word_vectors.save(DATA_MODELS_PATH+"word2vec")

#génération de X_train
X_train, train_time = Word2Vec.get_matrix_features_means(array_token, model)
pickle.dump(X_train, open(DATA_MODELS_PATH+'X_train.pkl','wb'))