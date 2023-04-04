import os
import sys
import pandas as pd
import numpy as np
import dill 
from src.exception import CustomException
from src.logger import logging 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
        dir_path= os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)          #dill is used to create a pkl file 

    except Exception as e:
        raise CustomException(e,sys)
    
