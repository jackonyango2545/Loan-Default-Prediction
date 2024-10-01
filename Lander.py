import streamlit as st 
from Pages import Model, Executive_Report, Home
import pandas as pd
import statistics
import numpy as np

@st.cache_resource
def data():  
    try:
        df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Jack\Loan_Defaultment\Loan_default.csv")
    except:
        df = pd.read_csv("https://github.com/jackonyango2545/Loan-Default-Prediction/raw/main/Loan_default.csv")
        
    df.rename(columns={'Income':'Annual Income','NumCreditLines':'Number of Credit Lines','DTIRatio':'Debt to Income Ratio'},inplace=True)

    #Lets convert the datatypes of some columns to minimise the space being used by the program
    df['Age'] = df['Age'].astype(np.int8)
    df['Annual Income'] = df['Annual Income'].astype(np.float64)
    df['LoanAmount'] = df['LoanAmount'].astype(np.float64)
    df['Default'] = df['Default'].astype('category')
    df['LoanID'] = df['LoanID'].astype('category')

    #Lets create the loan to income ratio, this will be used to see if the ratio matters 
    #Since we have diverse income and diverse loan amount

    #Lets create a new column that can be understood easily for defaultment
    def changer(x):
        if x == '1':
            return 'Yes'
        elif x == '0':
            return 'No'
        else:
            return x

    df['Loan Defaulted?'] = df['Default'].astype(str).apply(changer)
    del changer
    return df

#I want to use this to try to split time used in the Model page and also a section of the dataset can be used on the Executive Report
if 'df' not in st.session_state:
    st.session_state['df'] = data()

PAGES = {
    "Home": Home,
    "Executive Report": Executive_Report,
    "Model": Model
}

#st.write(st.session_state['df'])

selection = st.sidebar.radio("Navigate to", list(PAGES.keys()))

page = PAGES[selection]
page.app()