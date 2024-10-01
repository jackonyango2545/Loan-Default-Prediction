import streamlit as st



def app():
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
    
    st.title("LOAN DEFAULT PREDICTION EXECUTIVE REPORT")
    st.write("""
    **Problem Statement and Objective**\n
    Find out what factors tend to have an influence over loan defaulting and come up with a model that can be used by the 
    lending institutions to predict people who can fail to repay their loan
    """)

    st.write("""
    **Research Questions**\n
    - Does someone's age have a relationship with the loan default status? 
    - Is there a relationship between income, loan amount, and default status? 
    - Is there a relationship between Debt-to-Income Ratio, Loan-to-Income Ratio, and loan default status? 
    - Does employment type and months employed affect the default status? 
    - Does someone’s education level relate to the default status? 
    - Is there a relationship between marital status, dependents, and default status? 
    - Is there a relationship between interest rates, loan terms, and loan amounts? 
    - Is there a relationship between interest rate and default status? 
    - Is there a relationship between Credit score and interest rate? 
    - Is there a relationship between credit score, number of credit cards, and loan status? 
    - Do people who have mortgages tend to perform better than people who don’t in terms of paying loans? 
    - Does Having a cosigner tend to have an effect over someone's loan default?
        """)

    df = st.session_state['df']
    st.write("""
    **About the Data**\n
    The data used in this analysis was obtained from a secondary source, Kaggle. This dataset was chosen 
    due to its comprehensive features, which are instrumental in addressing our research questions and 
    solving the problem at hand. Additionally, the dataset’s recent update and the substantial number of records
    it contains further justify its selection.
    """,df.head(5))

    st.write("""
    **Result**\n
    - **Age**:Younger individuals (below 40) are more likely to default on loans than older individuals. 
    - **Income and Loan Amount**: Lower income and higher loan amounts are associated with increased default risk. 
    - **Debt-to-Income Ratio**: Higher debt-to-income ratios are linked to a higher likelihood of default. 
    - **Employment**: Full-time employment is associated with lower default rates, while unemployment increases the risk of default.
    Longer employment duration correlates with higher loan amounts and better repayment.
    - **Education**: Higher education levels are generally associated with lower default rates.
    - **Marital Status and Dependents**: Married individuals and those with dependents tend to have lower default rates.
    - **Interest Rates**: Lower interest rates are associated with lower likelyhood of default.
    - **Credit Score**: Credit scores have a negligible relationship with interest rates and loan default.
    - **Mortgages**: Individuals with mortgages tend to have better repayment performance.
    - **Cosigners**: Having a cosigner can positively impact loan repayment.
    """)

    st.write("""
    **Machine Learning Model**\n
    a. **Data Preprocessing**\n
    - Split the data into training (75%) and testing (25%) sets.
    - Encoded categorical columns.

    b. **Model**\n
    1. **Logistic Regression**: Initially used due to the nature of the problem. The model was trained and tested in two instances:
    - Before hyperparameter tuning: Achieved an accuracy of 61.248%.
    - After hyperparameter tuning: Slight improvement to 62.21% accuracy.
    - Due to its low performance, another model was considered for comparison.

    2. **XGBoost (Extreme Gradient Boosting)**: Chosen for its efficiency and high accuracy in handling large datasets.
    - Before hyperparameter tuning: Achieved an accuracy of 90.8783%.
    - After hyperparameter tuning: Slight decrease in performance to 88.643% accuracy.
    - Despite the slight decrease, XGBoost was selected due to its superior performance compared to logistic regression.
    """)


    st.write("""
    **Recommendations**\n
    - **Target specific demographics** :Focus on older age groups and individuals with higher incomes and lower debt-to-income ratios.
    - **Promote financial literacy** :Educate borrowers about responsible debt management and the consequences of default.
    - **Consider employment stability** :Evaluate employment history and job security when assessing loan applicants.
    - **Offer financial counseling** :Provide support to borrowers struggling with debt to help them improve their financial situation.
    - **Utilize credit scoring** :Use credit scores as a valuable tool in assessing creditworthiness.
    - **Consider cosigners** :Encourage borrowers to find cosigners to improve their chances of approval and reduce default risk.
    - **Set a maximum loan limit** :Limit loans to at most 2.5 times of the borrower’s annual income to ensure repayment capability.
    """)

    st.write("""
    **Limitations**\n
    - **Data Reliability:** Since the data is obtained from a secondary source (Kaggle), there may be biases related to the location and 
    why the data was collected. This could affect the generalizability of the findings.
    - **Dormain Knoledge:** Due to my background not being in finance or the credit industry, I may lack some essential skills 
    and insights that could improve the performance of the machine learning model. This limitation could impact the depth of the analysis 
    and the effectiveness of the recommendations and model.
    - **Data Imbalance:** The defaultment status has minority and majority classes which can lead to models under performance
    """)

