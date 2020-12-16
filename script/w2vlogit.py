import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as smet

CLEAN_PATH = "/home/bennis/Bureau/5GMM/AI-Frameworks/DefiIA/data/cleaned/"
MODEL_PATH = "/home/bennis/Bureau/5GMM/AI-Frameworks/DefiIA/data/models/"

Y = pd.read_csv(CLEAN_PATH+'train_label.csv')
''''enlever 5000'''
Y = Y['Category']
X = pickle.load( open( MODEL_PATH+"X_train.pkl", "rb" ) )

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=2020)


lr = LogisticRegression()
lr.fit(X_train, y_train)
print("score : ", lr.score(X_train,y_train))

pred = lr.predict(X_test)

print("f1score " , smet.f1_score(pred ,y_test, average='macro'))