import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle


# Load the trained model and preprocessing objects
model = tf.keras.models.load_model('ann_model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One hot encode the geography
geography_encoded = onehot_encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Concat the original data with the one hot encoded geography
input_data_final = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data_final)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
