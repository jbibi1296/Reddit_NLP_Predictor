class JackModelPipeline:
    def __init__(self,X,y):
        self.y = y
        self.X = X
        self.vec_params={}
        self.clf_params={}
        self.clasf = []
        self.clasf_string = []
        self.vecs = []
        self.vecs_string = []
        
    def import_libraries(self):
        import requests
        from bs4 import BeautifulSoup
        import time
        import pandas as pd
        import numpy as np
        import requests.auth
        import datetime
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score,recall_score,precision_score, confusion_matrix
        from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier, RandomForestClassifier
        from sklearn.svm import SVC
        import re
        from nltk.tokenize import punkt
        
    def add_clf(self,clf,string,params):
        self.clasf.append(clf)
        self.clasf_string.append(string)
        self.clf_params[string]=params  
            
    def add_vec(self,vec,string,params):
        self.vecs.append(vec)
        self.vecs_string.append(string)
        self.vec_params[string]=params  
            
    def show_clf_list(self):
        return self.clasf,self.clasf_string,self.clf_params

    def show_vec_list(self):
        return self.vecs,self.vecs_string,self.vec_params
    
    def show_params(self):
        return self.vec_params,self.clf_params
    
    def run_pipe(self):
        
        lists = []
        import sklearn.model_selection.train_test_split as train_test_split
        import sklearn.model_selection.GridSearchCV as GridSearchCV
        
        X_train, X_test, y_train, y_test = train_test_split(self.X,self.y, stratify = y,test_size=.2)
        
        number = 0
        for num_vec,vectorizer in enumerate(self.vecs):
            for num_clas,classifier in enumerate(self.clasf):

                #Create an empty dictionary to add to the dataframe
                dic = {}
                pipe_params = {}
                
                # Chooses the parameters based on whichever classifiers and vectorizers are being passed through
                vec_dict = self.vec_params[self.vecs_string[num_vec]]
                clasf_dict = self.clf_params[self.clasf_string[num_clas]]
                pipe_params.update(vec_dict)
                pipe_params.update(clasf_dict)

                # Create the Pipeline 
                pipe = Pipeline([(f'{self.vecs_string[num_vec]}',vectorizer),
                                 ((f'{self.clasf_string[num_clas]}',classifier))
                                ])
                
                # Set up the Grid Search
                gridvec = GridSearchCV(pipe,
                                       param_grid = pipe_params,
                                       cv=3,
                                       n_jobs = 5,
                                       verbose = 1)
                
                # Fit the GridSearch
                gridvec.fit(X_train,y_train)
                
                #Take the Best estimator
                bestest = gridvec.best_estimator_

                # Fit and transform the data on the best estimator
                bestest.fit(X_train,y_train)
                
                # save the classifier name, vectorizer name, and best parameters
                dic['classifier'] = self.clasf_string[num_clas]
                dic['vectorizer'] = self.vecs_string[num_vec]
                dic['Best_Params'] = gridvec.best_params_

                # Create the predictions for Y Train and Test data
                train_preds = bestest.predict(X_train)
                test_preds = bestest.predict(X_test)
                
                # Find the accuracy, precision, recall (Sensitivity) and specificity for both the Train Data and Test Data
                dic['train_accuracy'] = accuracy_score(y_train,train_preds)
                dic['test_accuracy'] = accuracy_score(y_test,test_preds)
                
                dic['train_precision'] = precision_score(y_train,train_preds,average='weighted')
                dic['test_precision'] = precision_score(y_test,test_preds,average='weighted')
                
                dic['train_recall'] = recall_score(y_train,train_preds,average='weighted')
                dic['test_recall'] = recall_score(y_test,test_preds,average='weighted')
                
                tn,fp,fn,tp  = confusion_matrix(y_train,train_preds).ravel()
                dic['train_specificity'] = tn/(tn+fp)
                tn,fp,fn,tp  = confusion_matrix(y_test,test_preds).ravel()
                dic['test_specificity'] = tn/(tn+fp)

                # Add the dictionary to a list
                lists.append(dic)
                
                # Let you know where you are up to 
                number +=1
                print(f'Done with {number} out of {(len(self.vecs)*len(self.clasf))} models.')

        # Transform the list to a Data Frame
        self.clf_df = pd.DataFrame(data = lists)
    def ShowGrid(self):
        return self.clf_df

        