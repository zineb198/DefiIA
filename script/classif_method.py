from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import os


class classif_method:
    '''
    This class applies different classification methods to X and Y data and adds metrics to a dataframe score_track
    '''

    def __init__(self, X_train, X_test_submit, Y_train, DATA_PATH_RESULTS):
        self.X_train = X_train
        self.X_test_submit = X_test_submit
        self.Y_train = Y_train
        self.data_path_results = DATA_PATH_RESULTS
        self.score_track = pd.DataFrame(columns=['name', 'f1score', 'accuracy', 'time'])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.Y_train, test_size=0.33,
                                                                                random_state=2020)

    def logit(self, save):
        lr = LogisticRegression(max_iter=1000)
        ts = time.time()
        lr.fit(self.X_train, self.y_train)
        te = time.time()
        pred = lr.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Logistic Regression',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = lr.predict(self.X_test_submit)
            self.to_submit_file('logit', pred_submit, self.data_path_results)

    def logit_lasso(self, save):
        logit_lasso = LogisticRegression(penalty='l1', solver='liblinear')
        ts = time.time()
        logit_lasso.fit(self.X_train, self.y_train)
        te = time.time()
        pred = logit_lasso.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Lasso Logistic Regression',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = logit_lasso.predict(self.X_test_submit)
            self.to_submit_file('logit_lasso', pred_submit, self.data_path_results)

    def logit_ridge(self, save):
        logit_ridge = LogisticRegression(penalty='l2', solver='liblinear')
        ts = time.time()
        logit_ridge.fit(self.X_train, self.y_train)
        te = time.time()
        pred = logit_ridge.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Ridge Logistic Regression',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = logit_ridge.predict(self.X_test_submit)
            self.to_submit_file('logit_ridge', pred_submit, self.data_path_results)

    def tree(self, save):
        tree = DecisionTreeClassifier()
        ts = time.time()
        tree.fit(self.X_train, self.y_train)
        te = time.time()
        pred = tree.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Regression Tree',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = tree.predict(self.X_test_submit)
            self.to_submit_file('tree', pred_submit, self.data_path_results)

    def forest(self, save):
        forest = RandomForestClassifier()
        ts = time.time()
        forest.fit(self.X_train, self.y_train)
        te = time.time()
        pred = forest.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Random Forest',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = forest.predict(self.X_test_submit)
            self.to_submit_file('forest', pred_submit, self.data_path_results)

    def Gradientboosting(self, save):
        gbm = GradientBoostingClassifier()
        ts = time.time()
        gbm.fit(self.X_train, self.y_train)
        te = time.time()
        pred = gbm.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Gradient Boosting',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = gbm.predict(self.X_test_submit)
            self.to_submit_file('gbm', pred_submit, self.data_path_results)

    def SVM(self, save):
        svm = SVC()
        ts = time.time()
        svm.fit(self.X_train, self.y_train)
        te = time.time()
        pred = svm.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Support Vector Machine',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = svm.predict(self.X_test_submit)
            self.to_submit_file('svm', pred_submit, self.data_path_results)

    def to_submit_file(self, method, pred_submit, DATA_PATH_RESULTS):
        self.X_test_submit["Category"] = pred_submit
        self.X_test_submit['Id'] = self.X_test_submit.index
        submit_file = self.X_test_submit[['Id', 'Category']]
        submit_file.to_csv(os.path.join(DATA_PATH_RESULTS, method), index=False)

    def method_save(self, method='all', save=True):
        if method == 'logit':
            self.logit(save)
        elif method == 'logit_lasso':
            self.logit_lasso(save)
        elif method == 'logit_ridge':
            self.logit_ridge(save)
        elif method == 'tree':
            self.tree(save)
        elif method == 'forest':
            self.forest(save)
        elif method == 'gbm':
            self.Gradientboosting(save)
        elif method == 'svm':
            self.SVM(save)

        elif method == 'all':
            self.logit(save)
            self.logit_lasso(save)
            self.logit_ridge(save)
            self.tree(save)
            self.forest(save)
            self.Gradientboosting(save)
            self.SVM(save)
        else:
            print('Try with method=logit or logit_lasso or logit_ridge or tree, etc')
