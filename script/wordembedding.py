import time
import gensim
import numpy as np
import hashlib
import pickle


class WordEmbedding:

    def __init__(self, word_embedding_type, args, DATA_MODELS_PATH):
        self.word_embedding_type = word_embedding_type
        self.args = args
        self.data_models_path = DATA_MODELS_PATH + self.word_embedding_type + '/'

    def train(self):
        print('### Training model ###')
        ts = time.time()
        if self.word_embedding_type == "word2vec":
            model = gensim.models.Word2Vec(**self.args)
        if self.word_embedding_type == "fasttext":  # Pas encore teste
            model = gensim.models.FastText(**self.args)
        te = time.time()
        return model, te - ts

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
        print('### Get features ###')
        ts = time.time()
        X_embedded_ = list(map(lambda x: WordEmbedding.get_features_mean(x, model), X))
        X_embedded = np.vstack(X_embedded_)
        te = time.time()
        return X_embedded, te - ts

    def get_hash(self): # surement a supprimer
        params = self.args.copy()
        del params['sentences']
        str_hash = ''
        for cle, valeur in params.items():
            str_hash += cle + '_' + str(valeur) + '_'
        return hashlib.md5(str_hash.encode()).hexdigest()

    def get_str(self):
        params = self.args.copy()
        del params['sentences']
        str_params = ''
        for cle, valeur in params.items():
            str_params += cle + '_' + str(valeur) + '_'
        return str_params

    def model_save(self, model, time):
        print("Model ", self.word_embedding_type, " trained in %.2f minutes" % (time / 60), "\n\n")
        num_hash = self.get_str()
        model.wv.save(self.data_models_path + num_hash)

    def features_save(self, X_train):
        num_hash = self.get_str()
        pickle.dump(X_train, open(self.data_models_path + num_hash + '.pkl', 'wb'))

    def train_save(self):
        model, time = self.train()
        self.model_save(model, time)
        X_train, time = self.get_matrix_features_means(self.args['sentences'], model)
        self.features_save(X_train)