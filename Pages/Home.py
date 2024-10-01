import streamlit as st

def app():
    st.title("LOAN DEFAULTMENT PREDICTION")
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

    st.write("""
        **Introduction**
        - When credit or financial institutions decide to lend money to individuals or businesses, they often face the risk of borrowers defaulting on their loans, leading to financial losses. 
        In this project, I use various statistical modeling techniques to analyze and understand trends that can predict the likelihood of someone defaulting on their loans.
        - Based on this analysis, I developed a model that can be used to make these predictions.
        """)

    st.write("""
        **Justification**
        - **Project Justification:** I chose this project because I believe it can be beneficial to banks and other financial institutions to understand the factors that can cause them to face losses and also be able to make sound and informed decisions to atleast minimise the lossess incurred.
        I believe with a few adjustments, It can proof to be beneficial to both the clients and the lending institutions
        - **Dataset Justification:** This dataset was chosen due to its rich and diverse features, which are crucial for analyzing and predicting loan default risks. 
        The comprehensive nature of the dataset allows for a thorough examination of various factors that influence loan repayment behavior. Additionally, since the dataset was updated recently, it captures new trends that can be used to understand the chances of someone defaulting on their loans.
        - **Data collection** The dataset used in this project was sourced from Kaggle, a reputable platform for data science and machine learning datasets. I chose to obtain it from a secondary source because primary data collection is expensive and time consuming, especially when we need a big data.""")