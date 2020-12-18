import pickle
import pandas as pd
from classification_fun import Classification_Fun

CLEAN_PATH = "C:/Users/Zineb BENISS/Desktop/insa 5GMM/AIF/DefiIA/data/cleaned/"
MODEL_PATH = "C:/Users/Zineb BENISS/Desktop/insa 5GMM/AIF/DefiIA/data/models/"

Y = pd.read_csv(CLEAN_PATH+'train_label.csv')
Y = Y['Category']
X = pickle.load( open( MODEL_PATH+"X_train.pkl", "rb" ) )
X = X

score_track = pd.DataFrame(columns = ['name', 'f1score', 'accuracy', 'time'])

cf = Classification_Fun(X, Y, score_track)

cf.logit()
cf.logit_lasso()
cf.logit_ridge()
cf.tree()
cf.forest()
cf.Gradientboosting()
cf.SVM()

#à voir que faire avec le dataframe (sauvegarder? si oui ou? avec le nom du model?)
print(cf.score_track)

