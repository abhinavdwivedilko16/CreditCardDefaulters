import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 ID:int ,
                LIMIT_BAL:float,
                AGE:int,
                BILL_AMT1:float,
                BILL_AMT2:float,
                BILL_AMT3:float,
                BILL_AMT4:float,
                BILL_AMT5:float,
                BILL_AMT6:float,
                PAY_AMT1:float,
                PAY_AMT2:float,
                PAY_AMT3:float,
                PAY_AMT4:float,
                PAY_AMT5:float,
                PAY_AMT6:float
                SEX:int,
                EDUCATION:int,
                MARRIAGE:int,
                PAY_0:int,
                PAY_2:int,
                PAY_3:int,
                PAY_4:int,
                PAY_5:int,
                PAY_6:int):
    self.ID=ID ,
    self.LIMIT_BAL=LIMIT_BAL,
    self.AGE=AGE,
    self.BILL_AMT1=BILL_AMT1,
    self.BILL_AMT2=BILL_AMT2,
    self.BILL_AMT3=BILL_AMT3,
    self.BILL_AMT4=BILL_AMT4,
    self.BILL_AMT5=BILL_AMT5,
    self.BILL_AMT6=BILL_AMT6,
    self.PAY_AMT1=PAY_AMT1,
    self.PAY_AMT2=PAY_AMT2,
    self.PAY_AMT3=PAY_AMT3,
    self.PAY_AMT4=PAY_AMT4,
    self.PAY_AMT5=PAY_AMT5,
    self.PAY_AMT6=PAY_AMT6,
    self.SEX=SEX,
    self.EDUCATION=EDUCATION,
    self.MARRIAGE=MARRIAGE,
    self.PAY_0=PAY_0,
    self.PAY_2=PAY_2,
    self.PAY_3=PAY_3,
    self.PAY_4=PAY_4,
    self.PAY_5=PAY_5,
    self.PAY_6=PAY_6
    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)