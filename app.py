import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load('income_model.pkl')
scaler = joblib.load('scaler.pkl')

mappings = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex': ['Male', 'Female'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
}

st.title('Income Prediction App')
st.write('Enter details to predict if income exceeds $50K/year')

with st.form('prediction_form'):
    age = st.number_input('Age', min_value=17, max_value=90, value=30)
    workclass = st.selectbox('Workclass', mappings['workclass'])
    fnlwgt = st.number_input('Final Weight', min_value=10000, max_value=1500000, value=200000)
    education = st.selectbox('Education', mappings['education'])
    education_num = st.number_input('Education Number', min_value=1, max_value=16, value=13)
    marital_status = st.selectbox('Marital Status', mappings['marital-status'])
    occupation = st.selectbox('Occupation', mappings['occupation'])
    relationship = st.selectbox('Relationship', mappings['relationship'])
    race = st.selectbox('Race', mappings['race'])
    sex = st.selectbox('Sex', mappings['sex'])
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=5000, value=0)
    hours_per_week = st.number_input('Hours per Week', min_value=1, max_value=99, value=40)
    native_country = st.selectbox('Native Country', mappings['native-country'])
    submitted = st.form_submit_button('Predict')

if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    le = LabelEncoder()
    for col in ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']:
        le.fit(mappings[col])
        input_data[col] = le.transform(input_data[col])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f'Predicted Income: >$50K (Probability: {probability:.2f})')
    else:
        st.warning(f'Predicted Income: <=$50K (Probability: {1-probability:.2f})')