import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.markdown(
    """
    <h1 style='text-align: center;'>📉 Customer Churn Prediction</h1>
    <p style='text-align: center; color: gray;'>
    Predict whether a customer is likely to leave based on key attributes
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("🧾 Customer Information")

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider(
    "Age",
    18, 92, 35
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    0, 10, 5
)

num_of_products = st.sidebar.slider(
    "Number of Products",
    1, 4, 2
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.sidebar.selectbox(
    "Is Active Member",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


st.subheader("📊 Customer Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Age", age)
col2.metric("Tenure", f"{tenure} yrs")
col3.metric("Products", num_of_products)


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

input_data_scaled = scaler.transform(input_data)


prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

st.subheader("🔮 Prediction Result")

st.progress(int(prediction_proba * 100))

st.metric(
    label="Churn Probability",
    value=f"{prediction_proba:.2%}"
)

if prediction_proba > 0.5:
    st.error("⚠️ This customer is likely to churn.")
else:
    st.success("✅ This customer is unlikely to churn.")

st.markdown(
    """
    <hr>
    <p style='text-align:center; color: gray;'>
    Built with Streamlit & TensorFlow
    </p>
    """,
    unsafe_allow_html=True
)
