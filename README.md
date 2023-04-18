## Welcome to Credit card Defaulters prediction

----------------------------------------------------------------------------------

# Problem Statement:
Financial threats are displaying a trend about the credit risk of commercial banks as the
incredible improvement in the financial industry has arisen. In this way, one of the
biggest threats faces by commercial banks is the risk prediction of credit clients. The
goal is to predict the probability of credit default based on credit card owner's
characteristics and payment history.

You have to build a solution that should able to predict the probability of credit
default based on credit card owner’s characteristics and payment history.

-------------------------------------------------------------------------------------------

## Dataset link: 
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

----------------------------------------------------------------------------------------------

# Content
## There are 25 variables:

### ID: ID of each client
### LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
### SEX: Gender (1=male, 2=female)
### EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
### MARRIAGE: Marital status (1=married, 2=single, 3=others)
### AGE: Age in years
### PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
### PAY_2: Repayment status in August, 2005 (scale same as above)
### PAY_3: Repayment status in July, 2005 (scale same as above)
### PAY_4: Repayment status in June, 2005 (scale same as above)
### PAY_5: Repayment status in May, 2005 (scale same as above)
### PAY_6: Repayment status in April, 2005 (scale same as above)
### BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
### BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
### BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
### BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
### BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
### BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
### PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
### PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
### PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
### PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
### PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
### PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
### default.payment.next.month: Default payment (1=yes, 0=no)

-------------------------------------------------------------------------------------------

## Software and Account requirement:

### 1)Github Account
### 2)VS Code IDE
### 3)GIT cli
### 4)GIT Documentation

-------------------------------------------------------------------------------------------------

## Creating an environment
```
conda create -p venv python==3.8 -y
```
## activating the environment
```
conda activate venv/
```
## Installing the requirements.txt
```
pip install -r requirements.txt
```
## Running the app.py
```
python app.py
```
### ->Click on the link generated and follow the webpage

### ->Enter all the details and click on Prediction

### ->output will be obtained, wheather a person is Defaulter or He is not a Defaulter.

-------------------------------------------------------------------------------
## Models that are being implemented and among these the best model will be selected: In this case SVM is being selected as it has the maximum accuracy.

### ->Linear Models-: Logistic Regression , Support Vector Machines

### ->Non-linear Models-:K-Nearest Neighbours ,Kernel SVM ,Naïve Bayes, Decision Tree Classification, Random Forest Classification

---------------------------------------------------
