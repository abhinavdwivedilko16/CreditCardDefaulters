import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test= (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1],

            )
            
            models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB()
            }
            params={
                "Decision Tree": {
                    #'criterion':[“gini”]
                    #'splitter' : [“best”, “random”],
                    #'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    #'criterion':[“gini”],
                 
                    #'max_features':['sqrt','log2'],
                    #'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{
                    #'penalty': ['l2',]
                    
                },
                "KNN":{
                    'n_neighbors':[5,10,20,25],
                    #'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "SVM":{
                    'gamma': ['auto'],
                },
                "Naive Bayes":{
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
        
        ## to get the best score from dict
            best_model_score = max(sorted(model_report.values()))

        ## to get the best model name from the dictionary
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]
            #print(best_model, best_model_name, best_model_score)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            ## train and evaluate the best model
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            return {"accuracy": accuracy} #{"classification_report": report}
        

        except Exception as e:
            raise CustomException(e,sys)