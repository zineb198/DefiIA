import time
import gensim
import numpy as np
import hashlib
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


class WordEmbedding:

    def __init__(self, word_embedding_type, args, DATA_MODELS_PATH, array_token, test_array_token):
        self.word_embedding_type = word_embedding_type
        self.args = args
        self.data_models_path = os.path.join(DATA_MODELS_PATH, self.word_embedding_type)
        self.array_token = array_token
        self.test_array_token = test_array_token

    def train(self):
        ts = time.time()
        if self.word_embedding_type == "word2vec":
            model = gensim.models.Word2Vec(**self.args)
        if self.word_embedding_type == "fasttext":  # Pas encore test
            model = gensim.models.FastText(**self.args)
        te = time.time()
        return model, te - ts

    @staticmethod
    def get_features_mean(lines, model):
        features = [model[x] for x in lines if x in model]
        if features==[]:
            fm = np.zeros(model.vector_size)
        else:
            fm = np.mean(features, axis=0)
        return fm

    @staticmethod
    def get_matrix_features_means(X, model):
        ts = time.time()
        X_embedded_ = list(map(lambda x: WordEmbedding.get_features_mean(x, model), X))
        X_embedded = np.vstack(X_embedded_)
        te = time.time()
        return X_embedded, te - ts

    @staticmethod
    def tokens_to_embedding_sequences(X, model):
        all_sequences_length = [len(x) for x in X]
        Ns = np.max(all_sequences_length)
        array_embedding_sequences = []
        for tokens in tqdm(X):
            embedding_sequence = []
            for token in tokens[:Ns]:
                embedding_sequence.append(model[token])
            array_embedding_sequences.append(embedding_sequence)
        X_embedded_pad = pad_sequences(array_embedding_sequences, value=0, maxlen=Ns, padding='pre')
        return X_embedded_pad

    def get_str(self):
        params = self.args.copy()
        del params['sentences']
        str_params = ''
        for cle, valeur in params.items():
            str_params += cle + '_' + str(valeur) + '_'
        return str_params

    def model_save(self, model, time):
        print("Model ", self.word_embedding_type, " trained in %.2f minutes" % (time / 60), "\n")
        num_hash = self.get_str()
        model.wv.save(os.path.join(self.data_models_path, num_hash))

    def features_save(self, X, file):
        num_hash = self.get_str()
        pickle.dump(X, open(os.path.join(self.data_models_path, file+'_'+num_hash+'.pkl'), 'wb'))

    def train_save(self, RNN=False):
        print('### Training model ###')
        print(self.get_str())
        model, time = self.train()
        self.model_save(model, time)

        if RNN:
            print('\n \n### Get features embedded padded for RNN with Keras ###')
            X_train = self.tokens_to_embedding_sequences(self.array_token, model)
            self.features_save(X_train, 'X_train_RNN')
            X_test = self.tokens_to_embedding_sequences(self.array_token, model)
            self.features_save(X_test, 'X_test_RNN')

        else:
            print('\n \n### Get features embedded ###')
            X_train, time = self.get_matrix_features_means(self.array_token, model)
            self.features_save(X_train, 'X_train')
            X_test, time = self.get_matrix_features_means(self.test_array_token, model)
            self.features_save(X_test, 'X_test')
