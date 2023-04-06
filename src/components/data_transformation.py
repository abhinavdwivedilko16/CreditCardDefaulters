# the main aim of data transformation is to do Feature Engineering, data cleaning , converting categorical features into numerical feat. etc
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #column transformer is used to create the pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass         #this class will help to capture the inputs for transformation
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):  #this function will create all my pickle files which will be responsible for transformation
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["ID",
                                 "LIMIT_BAL",
                                 "AGE",
                                 "BILL_AMT1",
                                 "BILL_AMT2",
                                 "BILL_AMT3",
                                 "BILL_AMT4",
                                 "BILL_AMT5",
                                 "BILL_AMT6",
                                 "PAY_AMT1",
                                 "PAY_AMT2",
                                 "PAY_AMT3",
                                 "PAY_AMT4",
                                 "PAY_AMT5",
                                 "PAY_AMT6"]
            
            categorical_columns = ["SEX",
                                   "EDUCATION",
                                   "MARRIAGE",
                                   "PAY_0",
                                   "PAY_2",
                                   "PAY_3",
                                   "PAY_4",
                                   "PAY_5",
                                   "PAY_6",
                                   ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="default.payment.next.month"

            numerical_columns = ["ID",
                                 "LIMIT_BAL",
                                 "AGE",
                                 "BILL_AMT1",
                                 "BILL_AMT2",
                                 "BILL_AMT3",
                                 "BILL_AMT4",
                                 "BILL_AMT5",
                                 "BILL_AMT6",
                                 "PAY_AMT1",
                                 "PAY_AMT2",
                                 "PAY_AMT3",
                                 "PAY_AMT4",
                                 "PAY_AMT5",
                                 "PAY_AMT6"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info( target_feature_train_df.shape)
            logging.info( target_feature_test_df.shape)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info( input_feature_train_arr.shape)
            logging.info( input_feature_test_arr.shape)

            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)