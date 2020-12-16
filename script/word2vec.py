import pandas as pd
import pickle
import numpy as np
import unicodedata
import re
import nltk
import time
import re
import unicodedata
import nltk
from tqdm import tqdm
import gensim



class Word2Vec:

    def __init__(self, args):
        self.args = args

    def train(self):
        ts = time.time()
        model = gensim.models.Word2Vec(**self.args)
        te = time.time()
        return model, te-ts

    @staticmethod
    def get_features_mean(lines, model):
        features = [model[x] for x in lines if x in model]
        if features == []:
            fm = np.zeros(model.vector_size)
        else:
            fm = np.mean(features, axis=0)
        return fm

    @staticmethod
    def get_matrix_features_means(X, model):
        ts = time.time()
        X_embedded_ = list(map(lambda x: Word2Vec.get_features_mean(x, model), X))
        X_embedded = np.vstack(X_embedded_)
        te = time.time()
        return X_embedded, te-ts