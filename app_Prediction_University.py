import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import streamlit as st
import plotly.express as px

st.write("""
# University Prediction App
This App predicts whether you will get admitted in your dream Universities!

Data obtained from the [Admission_Predict library](https://github.com/omkar04gaikwad/UinversityPredictions/blob/master/Admission_Predict.csv)
""")
st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input File](https://github.com/omkar04gaikwad/UinversityPredictions/blob/master/University_Prediciton_example.csv)
""")
#collects user input features into dataframe
file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
n = st.sidebar.slider('No. of rows in the uploaded CSV file', 1, 100, 1)
if file is not None:
    input_df = pd.read_csv(file)
else:
    def user_input_features():
        researc = st.sidebar.selectbox('Have you done any Research?', ('Yes', 'No'))
        if researc == 'Yes':
            research = 1
        elif researc == 'No':
            research = 0
        GRE = st.sidebar.slider('Your GRE score(out of 340)', 1.0, 340.0, 321.0)
        TOEFL = st.sidebar.slider('Your TOEFL score(out of 120)', 1.0, 120.0, 105.0)
        URATE = st.sidebar.slider('The University Rating you wan to apply(on the scale of 0 to 5)', 1.0, 5.0,4.0)
        SOP = st.sidebar.slider('Your SOP score(on the scale of 0 to 5)', 1.0, 5.0, 4.5)
        LOR = st.sidebar.slider('Your LOR score(on the scale of 0 to 5)', 1.0, 5.0, 3.5)
        CGPA = st.sidebar.slider('Your CGPA score(out of 10)', 1.0, 10.0, 8.50,)
        data = {'GRE Score': GRE,
                'TOEFL Score': TOEFL,
                'University Rating': URATE,
                'SOP': SOP,
                'LOR': LOR,
                'CGPA': CGPA,
                'Research': research}
        features = pd.DataFrame(data, index=[1])
        return features
    input_df = user_input_features()
raw = pd.read_csv('Admission_Predict.csv')
prediction_raw = raw.drop(columns=['Chance of Admit', 'Serial No.'])
df = pd.concat([input_df, prediction_raw], axis=0)
df = df[:n]
st.subheader('User Input features')

if file is not None:
    st.dataframe(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.dataframe(df)

model = pickle.load(open('University_Prediction.pkl', 'rb'))
predicts = model.predict(df)
#predicts_proba = model.predict_proba(df)

st.subheader('Prediction Output in Percentage')
st.table(pd.DataFrame({'Percentage':predicts*100}))

st.header('DATA VISUALISATION')
chart_data = raw
fig = px.scatter(chart_data, x='GRE Score', y='Chance of Admit', hover_data=['GRE Score', 'Chance of Admit'], height=700)
fig1 = px.scatter(chart_data, x='TOEFL Score', y='Chance of Admit', hover_data=['TOEFL Score', 'Chance of Admit'], height=700)
fig2 = px.scatter(chart_data, x='CGPA', y='Chance of Admit', hover_data=['CGPA', 'Chance of Admit'], height=700)
fig3 = px.scatter(chart_data, x='SOP', y='Chance of Admit', hover_data=['SOP', 'Chance of Admit'], height=700, width=700)
fig4 = px.scatter(chart_data, x='LOR', y='Chance of Admit', hover_data=['LOR', 'Chance of Admit'], height=700, width=700)
st.write(fig,fig1,fig2,fig3,fig4)
