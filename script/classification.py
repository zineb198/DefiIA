import pickle
import pandas as pd
import os
from classif_method import classif_method


# Define PATH files
DATA_CLEANED_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/cleaned/'
DATA_MODELS_PATH = '/Users/cecile/Documents/INSA/DefiIA/data/models/'

# Reading files
type_we = 'word2vec'
params_we = 'iter_10_sg_0_size_300_'

X_train = pickle.load(open(os.path.join(DATA_MODELS_PATH, type_we, 'train_' + params_we + '.pkl'), "rb"))
X_test = pickle.load(open(os.path.join(DATA_MODELS_PATH, type_we, 'test_' + params_we + '.pkl'), "rb"))
Y = pd.read_csv(DATA_CLEANED_PATH+'train_label.csv', index_col=0)
Y = Y['Category']


cf = classif_method(X_train, X_test, Y)

cf.logit()
#cf.logit_lasso()
#cf.logit_ridge()
#cf.tree()
#cf.forest()
#cf.Gradientboosting()
#cf.SVM()

#à voir que faire avec le dataframe (sauvegarder? si oui ou? avec le nom du model?)
print(cf.score_track)

