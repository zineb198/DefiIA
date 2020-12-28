import pickle
import pandas as pd
import os
from classif_method import classif_method
from wordembedding import WordEmbedding


# Define PATH files
#DATA_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/'  # TODO
DATA_PATH = '/home/cecile/data/'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)
DATA_MODELS_PATH = '/home/cecile/models/'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)
DATA_RESULTS_PATH = '/home/cecile/results/'  # PATH si utilisation de l'instance (attention il faut commenter les os.makedirs...)
DATA_CLEANED_PATH=os.path.join(DATA_PATH, 'cleaned')
DATA_PATH_TFIDF=os.path.join(DATA_MODELS_PATH, 'tf-idf')

#if not os.path.exists(DATA_MODELS_PATH): # Si utilisation de l'instance, il faut commenter les os.makedirs...
#    os.makedirs(DATA_MODELS_PATH)

#if not os.path.exists(os.path.join(DATA_MODELS_PATH, 'word2vec')):
#    os.makedirs(os.path.join(DATA_MODELS_PATH, 'word2vec'))
#    os.makedirs(os.path.join(DATA_MODELS_PATH, 'fasttext'))
   
#if not os.path.exists(os.path.join(DATA_MODELS_PATH, 'tf-idf')):
#    os.makedirs(os.path.join(DATA_MODELS_PATH, 'tf-idf'))
    
    
#if not os.path.exists(DATA_RESULTS_PATH):
#    os.makedirs(DATA_RESULTS_PATH)

# Reading files
params_cl = '_stem'  # TODO : '_stem' if stemming=True or '_lem' if lemmatizer in cleaning

arg_gender=False # TODO True if add gender to models, else False
embedding = True #TODO True if embedding, else if TF-IDF

train_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'train_cleaned' + params_cl + '.csv'), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_CLEANED_PATH, 'test_cleaned' + params_cl + '.csv'), index_col=0)

if embedding:
    # Reading files for each word embedding combination of parameters
    for sg in [0, 1]:
        for iter in [10]:
            type_we = 'word2vec'
            params_we = dict(sentences=None, iter=iter, sg=sg, size=300, min_count=1, window=5, hs=0, negative=10)

            we = WordEmbedding(word_embedding_type=type_we, args=params_we,
                               DATA_MODELS_PATH=DATA_MODELS_PATH, array_token=None,
                               test_array_token=None)
            name_we = we.get_str()
            print(name_we)

            X_train = pickle.load(open(os.path.join(DATA_MODELS_PATH, type_we, 'X_train_' + name_we + '.pkl'), "rb"))
            X_test_submit = pickle.load(open(os.path.join(DATA_MODELS_PATH, type_we, 'X_test_' + name_we + '.pkl'), "rb"))
            Y_train = pd.read_csv(DATA_PATH + 'train_label.csv', index_col=0)
            Y_train = Y_train['Category']

    # Apply classification method
            cf = classif_method(X_train, X_test_submit, Y_train, train_df, test_df, DATA_RESULTS_PATH, name_we,arg_gender)
            cf.method_save(save=True)

else :
    X_train = pickle.load(open(os.path.join(DATA_PATH_TFIDF,'X_train.pickle'), "rb"))
    X_test_submit = pickle.load(open(os.path.join(DATA_PATH_TFIDF,'X_test.pickle'), "rb"))
    Y_train = pd.read_csv(DATA_PATH + 'train_label.csv', index_col=0)
    Y_train = Y_train['Category']

    # Apply classification method
    cf = classif_method(X_train, X_test_submit, Y_train, train_df, test_df, DATA_RESULTS_PATH,'tf-idf_', arg_gender)
    cf.method_save(save=True)