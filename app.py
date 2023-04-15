from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)


## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdefaulter',methods=['GET','POST'])
def predict_defaulter():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            ID = int(request.form.get('ID')),
            LIMIT_BAL = float(request.form.get('LIMIT_BAL')),
            AGE = int(request.form.get('AGE')),
            BILL_AMT1 = float(request.form.get('BILL_AMT1')),
            BILL_AMT2 = float(request.form.get('BILL_AMT2')),
            BILL_AMT3 = float(request.form.get('BILL_AMT3')),
            BILL_AMT4 = float(request.form.get('BILL_AMT4')),
            BILL_AMT5 = float(request.form.get('BILL_AMT5')),
            BILL_AMT6 = float(request.form.get('BILL_AMT6')),
            PAY_AMT1 = float(request.form.get('PAY_AMT1')),
            PAY_AMT2 = float(request.form.get('PAY_AMT2')),
            PAY_AMT3 = float(request.form.get('PAY_AMT3')),
            PAY_AMT4 = float(request.form.get('PAY_AMT4')),
            PAY_AMT5 = float(request.form.get('PAY_AMT5')),
            PAY_AMT6 = float(request.form.get('PAY_AMT6')),
            SEX = int(request.form.get('SEX')),
            EDUCATION = int(request.form.get('EDUCATION')),
            MARRIAGE = int(request.form.get('MARRIAGE')),
            PAY_0 = int(request.form.get('PAY_0')),
            PAY_2 = int(request.form.get('PAY_2')),
            PAY_3 = int(request.form.get('PAY_3')),
            PAY_4 = int(request.form.get('PAY_4')),
            PAY_5 = int(request.form.get('PAY_5')),
            PAY_6 = int(request.form.get('PAY_6'))
        )


        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',result=result[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")