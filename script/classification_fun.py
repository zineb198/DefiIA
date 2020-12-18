from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


class Classification_Fun:
	'''
	This class applies different classification methods to X and Y data and 
	adds the result to a dataframe
	'''

	def __init__(self, X, Y, score_track = None):

		self.X = X
		self.Y = Y

		if score_track==None:
			self.score_track = pd.DataFrame(columns = ['name', 'f1score', 'accuracy', 'time'])
		else:
			self.score_track = score_track

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.33, random_state=2020)

	def logit(self):
		lr = LogisticRegression(max_iter=1000)
		ts = time.time()
		lr.fit(self.X_train, self.y_train)
		te = time.time()
		pred = lr.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Logistic Regression',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)
		

	def logit_lasso(self):
		logit_lasso = LogisticRegression(penalty='l1',solver='liblinear')
		ts = time.time()
		logit_lasso.fit(self.X_train,self.y_train)
		te = time.time()
		pred=logit_lasso.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Lasso Logistic Regression',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)

	def logit_ridge(self):
		logit_ridge = LogisticRegression(penalty='l2',solver='liblinear')
		ts = time.time()
		logit_ridge.fit(self.X_train,self.y_train)
		te = time.time()
		pred=logit_ridge.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Ridge Logistic Regression',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)

	def tree(self):
		tree= DecisionTreeClassifier()
		ts = time.time()
		tree.fit(self.X_train,self.y_train)
		te = time.time()
		pred=tree.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Regression Tree',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)

	def forest(self):
		forest = RandomForestClassifier()
		ts = time.time()
		forest.fit(self.X_train,self.y_train)
		te = time.time()
		pred=forest.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Random Forest',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)

	def Gradientboosting(self):
		gbm = GradientBoostingClassifier()
		ts = time.time()
		gbm.fit(self.X_train,self.y_train)
		te = time.time()
		pred=gbm.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Gradient Boosting',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)


	def SVM(self):
		svm= SVC()
		ts = time.time()
		svm.fit(self.X_train,self.y_train)
		te = time.time()
		pred=svm.predict(self.X_test)
		self.score_track = self.score_track.append({'name' : 'Support Vector Machine',
			'f1score' : f1_score(self.y_test,pred, average = 'macro'),
			 'accuracy' : accuracy_score(self.y_test,pred),'time' :te-ts}, ignore_index = True)


