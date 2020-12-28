from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import numpy as np
import os


class classif_method:
    '''
    This class applies different classification methods to X and Y data and adds metrics to a dataframe score_track
    
    train_df and test_df are dataframe with columns : Id, Category and gender
    gender = Boolean, if True add columns genders to predict
    '''

    def __init__(self, X_train, X_test_submit, Y_train, train_df, test_df, DATA_RESULTS_PATH, params_we_str,gender):
        if gender==False :
            self.X_train_init=X_train
            self.X_test_submit = X_test_submit
            self.Y_train_init = Y_train
        else :
            if type(X_train)==csr_matrix: #case TF-IDF
                gender_train=csr_matrix(pd.get_dummies(train_df[['gender']]))
                self.X_train_init=hstack((X_train, gender_train))
                gender_test=csr_matrix(pd.get_dummies(test_df[['gender']]))
                self.X_test_submit =hstack((X_test_submit, gender_test))
                self.Y_train_init = Y_train

            elif type(X_train)==np.ndarray: #case embedding
                gender_train=pd.get_dummies(train_df[['gender']]).values
                self.X_train_init=np.hstack((X_train, gender_train))
                gender_test=pd.get_dummies(test_df[['gender']]).values
                self.X_test_submit =np.hstack((X_test_submit, gender_test))
                self.Y_train_init = Y_train
            else : 
                print('Type error X_train')
                
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train_init,
                                                                                    self.Y_train_init,test_size=0.33,
                                                                                    random_state=2020)
        self.test_df = test_df 
        self.data_results_path = DATA_RESULTS_PATH
        self.params_we = params_we_str
        self.score_track = pd.DataFrame(columns=['name', 'f1score', 'accuracy', 'time'])

        

    def logit(self, save):
        if save==False :
            lr = LogisticRegression(max_iter=10000)
            ts = time.time()
            lr.fit(self.X_train, self.y_train)
            te = time.time()
            pred = lr.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Logistic Regression',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 2),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 2),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            lr = LogisticRegression(max_iter=10000)
            lr.fit(self.X_train_init, self.Y_train_init)
            pred_submit = lr.predict(self.X_test_submit)
            self.to_submit_file('logit', pred_submit)

    def logit_lasso(self, save):
        if save==False :
            logit_lasso = LogisticRegression(penalty='l1', solver='liblinear')
            ts = time.time()
            logit_lasso.fit(self.X_train, self.y_train)
            te = time.time()
            pred = logit_lasso.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Lasso Logistic Regression',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else :
            logit_lasso = LogisticRegression(penalty='l1', solver='liblinear')
            logit_lasso.fit(self.X_train_init, self.Y_train_init)
            pred_submit = logit_lasso.predict(self.X_test_submit)
            self.to_submit_file('logit_lasso', pred_submit)

    def logit_ridge(self, save):
        if save==False :
            logit_ridge = LogisticRegression(penalty='l2', solver='liblinear')
            ts = time.time()
            logit_ridge.fit(self.X_train, self.y_train)
            te = time.time()
            pred = logit_ridge.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Ridge Logistic Regression',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            logit_ridge = LogisticRegression(penalty='l2', solver='liblinear')
            logit_ridge.fit(self.X_train_init, self.Y_train_init)
            pred_submit = logit_ridge.predict(self.X_test_submit)
            self.to_submit_file('logit_ridge', pred_submit)

    def tree(self, save):
        if save==False:
            tree = DecisionTreeClassifier()
            gridsearch_tree = GridSearchCV(tree,param_grid={"max_depth":list(range(4,10))+[None]},scoring="f1_macro")
            ts = time.time()
            gridsearch_tree.fit(self.X_train, self.y_train)
            te = time.time()
            pred = gridsearch_tree.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Regression Tree',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            tree = DecisionTreeClassifier()
            gridsearch_tree = GridSearchCV(tree,param_grid={"max_depth":list(range(4,12))+[None]},scoring="f1_macro")
            gridsearch_tree.fit(self.X_train_init, self.Y_train_init)
            pred_submit = gridsearch_tree.predict(self.X_test_submit)
            self.to_submit_file('tree', pred_submit)

    def forest(self, save):
        if save==False:
            forest = RandomForestClassifier()
            param_dist ={"max_features":["sqrt","log2"],"n_estimators":list(range(100,500,1000)),
                         "max_depth":list(range(4,10))+[None]}
            random_search_rf = RandomizedSearchCV(forest,param_distributions=param_dist,
                                                  n_iter=20,scoring="f1_macro")
            ts = time.time()
            random_search_rf.fit(self.X_train, self.y_train)
            te = time.time()
            pred = random_search_rf.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Random Forest',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            forest = RandomForestClassifier()
            param_dist ={"max_features":{"sqrt","log2"},"n_estimators":list(range(100,500,1000)),
                         "max_depth":list(range(4,10))+[None]}
            random_search_rf = RandomizedSearchCV(forest,param_distributions=param_dist,
                                                  n_iter=10,scoring="f1_macro")
            random_search_rf.fit(self.X_train_init, self.Y_train_init)
            pred_submit = random_search_rf.predict(self.X_test_submit)
            self.to_submit_file('forest', pred_submit)

    def Gradientboosting(self, save):
        if save==False:
            param_dist = {'n_estimators': list(range(100,601,100)), 'learning_rate': [0.05,0.1,0.25,0.5,0.75,1]}
            gbm = GradientBoostingClassifier()
            random_search_gbm = RandomizedSearchCV(gbm,param_distributions=param_dist,n_iter=10,scoring="f1_macro")
            ts = time.time()
            random_search_gbm.fit(self.X_train, self.y_train)
            te = time.time()
            pred = random_search_gbm.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Gradient Boosting',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            param_dist = {'n_estimators': list(range(100,601,100)), 'learning_rate': [0.05,0.1,0.25,0.5,0.75,1]}
            gbm = GradientBoostingClassifier()
            random_search_gbm = RandomizedSearchCV(gbm,param_distributions=param_dist,n_iter=10,scoring="f1_macro")
            random_search_gbm.fit(self.X_train_init, self.Y_train_init)
            pred_submit = random_search_gbm.predict(self.X_test_submit)
            self.to_submit_file('gbm', pred_submit)

    def SVM(self, save):
        if save==False:
            param_dist = {"C":[0.001, 0.01, 0.1, 1, 10, 100],
                          'kernel': ['linear','poly','rbf'],
                          'gamma':['scale',0.001, 0.01, 0.1, 1, 10, 100]}
            svm = SVC()
            random_search_svm = RandomizedSearchCV(svc,param_distributions=param_dist,n_iter=10,scoring="f1_macro")
            ts = time.time()
            random_search_svm.fit(self.X_train, self.y_train)
            te = time.time()
            pred = random_search_svm.predict(self.X_test)
            self.score_track = self.score_track.append({'name': 'Support Vector Machine',
                                                        'f1score': round(f1_score(self.y_test, pred, average='macro'), 3),
                                                        'accuracy': round(accuracy_score(self.y_test, pred), 3),
                                                        'time': round(te - ts)}, ignore_index=True)
        else:
            param_dist = {"C":[0.001, 0.01, 0.1, 1, 10, 100],
                          'kernel': ['linear','poly','rbf'],
                          'gamma':['scale',0.001, 0.01, 0.1, 1, 10, 100]}
            svm = SVC()
            random_search_svm = RandomizedSearchCV(svc,param_distributions=param_dist,n_iter=10,scoring="f1_macro")
            random_search_svm.fit(self.X_train_init, self.Y_train_init)
            pred_submit = random_search_svm.predict(self.X_test_submit)
            self.to_submit_file('svm', pred_submit)

    def to_submit_file(self, method, pred_submit):
        test_df_copy=self.test_df.copy()
        test_df_copy["Category"]=pred_submit
        submit_file=test_df_copy[["Id","Category"]]
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
