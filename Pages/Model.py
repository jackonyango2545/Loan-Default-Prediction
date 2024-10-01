import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import streamlit as st


# We need to access the data we stored on the session state

def app():
    st.write("*This project is meant for learning purposes*")
    st.markdown("""
    <style>
    .main{
        background: linear-gradient(to right, #2F2DD2, #090B99);
        color: white;
    }
    .sidebar .sidebar-content .navbar{
        background: #0E08E3 !important;}
    </style>
    """, unsafe_allow_html=True)

    st.title("Loan Default Prediction Model")
    @st.cache_resource
    def data():
        df1 = st.session_state['df']

        df1['Default'] = df1['Default'].astype(bool)

        # Split the data into features and target variables
        x = df1[['Age', 'Annual Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'LoanTerm', 
            'Debt to Income Ratio', 'HasMortgage', 'HasDependents', 'HasCoSigner','EmploymentType','Education','MaritalStatus']]
        y = df1['Default']

        #First we need to preprocess the data such that changine the datatypes, encoding the data, and spliting the data into train test samples
        x['Age'] = x['Age'].astype(np.uint8)
        x['Annual Income'] = x['Annual Income'].astype(np.uint32)
        x['LoanAmount'] = x['LoanAmount'].astype(np.uint32)
        x['CreditScore'] = x['CreditScore'].astype(np.uint16)
        x['InterestRate'] = x['InterestRate'].astype(np.float16)
        x['LoanTerm'] = x['LoanTerm'].astype(np.uint16)
        x['Debt to Income Ratio'] = x['Debt to Income Ratio'].astype(np.float16)
        x['HasMortgage'] = x['HasMortgage'].astype(bool)
        x['HasDependents'] = x['HasDependents'].astype(bool)
        x['HasCoSigner'] = x['HasCoSigner'].astype(bool)
        x['MonthsEmployed'] = x['MonthsEmployed'].astype(np.int8)

        #Lets start by manual encoding of the data, we are going to use a function for this
        def change_encode(x):
            if x == 'Yes':
                return '1'
            elif x== 'No':
                return '0'
            else:
                return x

        x['HasMortgage'] = x['HasMortgage'].apply(change_encode)
        x['HasDependents'] = x['HasDependents'].apply(change_encode)
        x['HasCoSigner'] = x['HasCoSigner'].apply(change_encode)

        x['HasMortgage'] = x['HasMortgage'].astype(bool)
        x['HasDependents'] = x['HasDependents'].astype(bool)
        x['HasCoSigner'] = x['HasCoSigner'].astype(bool)


        #Lets use pandas library to encode the remaining columns
        try:
            x = pd.get_dummies(x,columns=['EmploymentType','Education','MaritalStatus'])
        except:
            exit
        
        #Lets split our dataset for train test split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.25, random_state=42)
        return Xtrain, Xtest, Ytrain, Ytest

    Xtrain, Xtest, Ytrain, Ytest = data()

    @st.cache_resource
    def model():
        xgb_classifier = xgb.XGBClassifier(objective='binary:logistic',colsample_bytree=1.0,gamma=0.2,reg_alpha=1,reg_lambda=1,learning_rate=0.05,min_child_weight=7,subsample=0.6, use_label_encoder=False, eval_metric='logloss')
        # Let us fit the model
        xgb_classifier.fit(Xtrain, Ytrain)
        y_pred_proba = xgb_classifier.predict_proba(Xtest)[:, 1]

        # Evaluate the model
        y_pred = xgb_classifier.predict(Xtest)
        accuracy = accuracy_score(y_pred,Ytest)
        return xgb_classifier, accuracy


    xgb_classifier,accuracy = model()
    st.write("""
    **Models Accuracy**\n
    The model has an accuracy level of""",round(accuracy*100,2),"percent")

    radiooptions = ['Yes','No']

    st.write("**Personal Information**")
    Age = st.number_input("Age", min_value=18, max_value=70)
    Marital_Status = st.radio("Marital Status", options=['Single', 'Married', 'Divorced'], horizontal=True)
    Dependents = st.radio("Do you have Dependents?", options=radiooptions, horizontal=True)

    st.write("**Financial Information**")
    Monthly_Income = st.number_input("Monthly Income", value=1000)
    Loan_Amount = st.number_input("Loan Amount", min_value=1)
    Loan_Term = st.number_input("Loan Term in Months", min_value=1, max_value=240, value=12)
    Debt = st.number_input("How much of your monthly salary goes to debt?", min_value=0)
    Mortgage = st.radio("Do you have mortgage?", options=radiooptions, horizontal=True)

    st.write("**Credit Information**")
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=850, value=575)
    Cosigner = st.radio("Do you have a guarantor or collateral", options=radiooptions, horizontal=True)

    if Credit_Score>=800:
        if Cosigner=="Yes":
            if Loan_Term<25:
                InterestRate=3
            else:
                InterestRate= 4
        else:
            if Loan_Term<25:
                InterestRate=3.5
            else:
                InterestRate= 5
    elif Credit_Score>=740:
        if Cosigner=="Yes":
            if Loan_Term<25:
                InterestRate=4.7
            else:
                InterestRate= 5.5
        else:
            if Loan_Term<25:
                InterestRate=5
            else:
                InterestRate= 7
    elif Credit_Score>=670:
        if Cosigner=="Yes":
            if Loan_Term<25:
                InterestRate=6.5
            else:
                InterestRate= 9
        else:
            if Loan_Term<25:
                InterestRate=7
            else:
                InterestRate= 10
    elif Credit_Score>=580:
        if Cosigner=="Yes":
            if Loan_Term<25:
                InterestRate=9
            else:
                InterestRate= 13
        else:
            if Loan_Term<25:
                InterestRate=10
            else:
                InterestRate= 15
    elif Credit_Score>=300:
        if Cosigner=="Yes":
            if Loan_Term<25:
                InterestRate=13
            else:
                InterestRate= 18
        else:
            if Loan_Term<25:
                InterestRate=16
            else:
                InterestRate= 20
    else:
        InterestRate= 0

    st.write("**Interest Rate is**",InterestRate)


    st.write("**Employment Information**")
    Employment_Type = st.radio("Employment Type", options=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], horizontal=True)
    Months_Employed = st.number_input("Months Employed", min_value=0)

    st.write("**Education Information**")
    Education = st.radio("Highest Education Attained", options=['High School', "Bachelor's", "Master's", 'PhD'], horizontal=True)

    Column_A, Column_B = st.columns([0.9,1])

    with Column_A:
        Loan_Payout = st.button("Loan Payment Structure")
        if Loan_Payout:
            st.write("Loan Amount",Loan_Amount)
            st.write("Loan Term in Months",Loan_Term)
            st.write("Interest Rate per Annum",InterestRate)

            #Lets calculate the interest earned and the final payout
            rate = 1+(InterestRate/1200)
            formulae = rate ** Loan_Term
            Amount = round((Loan_Amount * formulae),2)
            interest_earned = Amount - Loan_Amount

            st.write("Amount Owed",Amount)
            st.write("Interest Charged",round(interest_earned,2))
            st.write("Adviced Monthly Payments",round((Amount/Loan_Term),2))
        
    with Column_B:
        a = st.button("Predict")
        if a:
            def output():
                #Before we do our prediction, we need to first prepare our data
                Annual_Income= Monthly_Income * 12
                Debt_to_income_ratio= Debt / Monthly_Income
                            
                if Mortgage == "Yes":
                    b = '1'
                else:
                    b = '0'

                if Dependents == "Yes":
                    D='1'
                else:
                    D= '0'

                if Cosigner == "Yes":
                    C='1'
                else:
                    C= '0'
                            
                if Employment_Type == "Full-time":
                    FT = '1'
                    PT = '0'
                    SE = '0'
                    UE = '0'
                elif Employment_Type == "Part-time":
                    FT = '0'
                    PT = '1'
                    SE = '0'
                    UE = '0'
                elif Employment_Type == "Self-employed":
                    FT = '0'
                    PT = '0'
                    SE = '1'
                    UE = '0'
                elif Employment_Type == "Unemployed":
                    FT = '0'
                    PT = '0'
                    SE = '0'
                    UE = '1'

                if Education == 'High School':
                    EB = '0'
                    EH = '1'
                    EM = '0'
                    EP = '0'
                elif Education == "Bachelor's":
                    EB = '1'
                    EH = '0'
                    EM = '0'
                    EP = '0'
                elif Education == "Master's":
                    EB = '0'
                    EH = '0'
                    EM = '1'
                    EP = '0'
                elif Education == 'PhD':
                    EB = '0'
                    EH = '0'
                    EM = '0'
                    EP = '1'
                
                if Marital_Status == 'Single':
                    Single = '1'
                    Married = '0'
                    Divorced = '0'
                elif Marital_Status == 'Married':
                    Single = '0'
                    Married = '1'
                    Divorced = '0'
                elif Marital_Status == 'Divorced':
                    Single = '0'
                    Married = '0'
                    Divorced = '1'

                userData = pd.DataFrame(
                    {
                        'Age': [Age],
                        'Annual Income':[Annual_Income],
                        'LoanAmount':[Loan_Amount],
                        'CreditScore':[Credit_Score],
                        'MonthsEmployed':[Months_Employed],
                        'InterestRate':[InterestRate],
                        'LoanTerm':[Loan_Term],
                        'Debt to Income Ratio':[Debt_to_income_ratio],
                        'HasMortgage':[b],
                        'HasDependents':[D],
                        'HasCoSigner':[C],
                        'EmploymentType_Full-time':[FT],
                        'EmploymentType_Part-time':[PT],
                        'EmploymentType_Self-employed':[SE],
                        'EmploymentType_Unemployed':[UE],
                        "Education_Bachelor's":[EB],
                        'Education_High School':[EH],
                        "Education_Master's":[EM],
                        'Education_PhD':[EP],
                        'MaritalStatus_Divorced':[Divorced],
                        'MaritalStatus_Married':[Married],
                        'MaritalStatus_Single':[Single]
                    }
                )
                
                userData['HasMortgage'] = userData['HasMortgage'].astype(bool)
                userData['HasDependents'] = userData['HasDependents'].astype(bool)
                userData['HasCoSigner'] = userData['HasCoSigner'].astype(bool)
                userData['EmploymentType_Full-time'] = userData['EmploymentType_Full-time'].astype(bool)
                userData['EmploymentType_Part-time'] = userData['EmploymentType_Part-time'].astype(bool)
                userData['EmploymentType_Self-employed'] = userData['EmploymentType_Self-employed'].astype(bool)
                userData['EmploymentType_Unemployed'] = userData['EmploymentType_Unemployed'].astype(bool)
                userData["Education_Bachelor's"] = userData["Education_Bachelor's"].astype(bool)
                userData['Education_High School'] = userData['Education_High School'].astype(bool)
                userData["Education_Master's"] = userData["Education_Master's"].astype(bool)
                userData['Education_PhD'] = userData['Education_PhD'].astype(bool)
                userData['MaritalStatus_Divorced'] = userData['MaritalStatus_Divorced'].astype(bool)
                userData['MaritalStatus_Married'] = userData['MaritalStatus_Married'].astype(bool)
                userData['MaritalStatus_Single'] = userData['MaritalStatus_Single'].astype(bool)

                prediction = xgb_classifier.predict_proba(userData)[:, 1]
                default_probability = prediction[0]
                chance = round((default_probability*100),2)
                otherwise = ((1 - default_probability)*100)
                # Interpret the prediction
                if default_probability > 0.5:
                    message = f"There is a {chance:.2f}% chance of the loan being defaulted. Meaning there is {otherwise:.2f}% chance of default"
                elif default_probability < 0.5:
                    message = f"There is a {otherwise:.2f}% chance of the loan not being defaulted. Meaning there is {chance:.2f}% chance of no default"
                return message

            message=output()
            st.write(message)