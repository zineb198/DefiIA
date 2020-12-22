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

    def __init__(self, X_train, X_test_submit, Y_train, test_df, DATA_RESULTS_PATH, params_we_str):
        self.X_train = X_train
        self.X_test_submit = X_test_submit
        self.Y_train = Y_train
        self.test_df = test_df
        self.data_results_path = DATA_RESULTS_PATH
        self.params_we = params_we_str
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
            self.to_submit_file('logit', pred_submit)

    def logit_lasso(self, save):
        logit_lasso = LogisticRegression(penalty='l1', solver='liblinear')
        ts = time.time()
        logit_lasso.fit(self.X_train, self.y_train)
        te = time.time()
        pred = logit_lasso.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Lasso Logistic Regression',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = logit_lasso.predict(self.X_test_submit)
            self.to_submit_file('logit_lasso', pred_submit)

    def logit_ridge(self, save):
        logit_ridge = LogisticRegression(penalty='l2', solver='liblinear')
        ts = time.time()
        logit_ridge.fit(self.X_train, self.y_train)
        te = time.time()
        pred = logit_ridge.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Ridge Logistic Regression',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = logit_ridge.predict(self.X_test_submit)
            self.to_submit_file('logit_ridge', pred_submit)

    def tree(self, save):
        tree = DecisionTreeClassifier()
        ts = time.time()
        tree.fit(self.X_train, self.y_train)
        te = time.time()
        pred = tree.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Regression Tree',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = tree.predict(self.X_test_submit)
            self.to_submit_file('tree', pred_submit)

    def forest(self, save):
        forest = RandomForestClassifier()
        ts = time.time()
        forest.fit(self.X_train, self.y_train)
        te = time.time()
        pred = forest.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Random Forest',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = forest.predict(self.X_test_submit)
            self.to_submit_file('forest', pred_submit)

    def Gradientboosting(self, save):
        gbm = GradientBoostingClassifier()
        ts = time.time()
        gbm.fit(self.X_train, self.y_train)
        te = time.time()
        pred = gbm.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Gradient Boosting',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = gbm.predict(self.X_test_submit)
            self.to_submit_file('gbm', pred_submit)

    def SVM(self, save):
        svm = SVC()
        ts = time.time()
        svm.fit(self.X_train, self.y_train)
        te = time.time()
        pred = svm.predict(self.X_test)
        self.score_track = self.score_track.append({'name': 'Support Vector Machine',
                                                    'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                    'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                    'time': round(te - ts)}, ignore_index=True)
        if save:
            pred_submit = svm.predict(self.X_test_submit)
            self.to_submit_file('svm', pred_submit)

    def to_submit_file(self, method, pred_submit):
        self.test_df["Category"] = pred_submit
        self.test_df['Id'] = self.test_df.index
        submit_file = self.test_df[['Id', 'Category']]
        submit_file.to_csv(os.path.join(self.data_results_path, self.params_we + method + '.csv'), index=False)

    def method_save(self, method='all', save=False):
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
            print('### logit ###')
            self.logit(save)
            #print('### logit lasso ###')
            #self.logit_lasso(save)
            #print('### logit ridge ###')
            #self.logit_ridge(save)
            print('### tree ###')
            self.tree(save)
            print('### forest ###')
            self.forest(save)
            #print('### GBM ###')
            #self.Gradientboosting(save)
            #print('### SVM ###')
            #self.SVM(save)
            self.score_track.to_csv(os.path.join(self.data_results_path, self.params_we + '_all.csv'))
        else:
            print('Try with method=logit or logit_lasso or logit_ridge or tree, etc')
