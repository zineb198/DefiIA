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
#if not os.path.exists(DATA_RESULTS_PATH):
#    os.makedirs(DATA_RESULTS_PATH)

test_df = pd.read_json(DATA_PATH+"/test.json")
test_df.set_index('Id', inplace=True)
test_df['description'] = [line.replace('\r', '') for line in test_df["description"].values]

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
        cf = classif_method(X_train, X_test_submit, Y_train, test_df, DATA_RESULTS_PATH, name_we)
        cf.method_save(save=True)

