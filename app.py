import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(
    page_title='Bank Customer Churn Predictor',
    page_icon='app_icon.svg',
    # layout='wide',  # Changed from 'centered' to 'wide'
    initial_sidebar_state='auto',
)

# Load the trained model
def load_model():
    model_path = os.path.join('model', 'Bank_Churn_pred_model.joblib')
    return joblib.load(model_path)

model = load_model()

# Display the app icon and title
col1, col2 = st.columns([2, 15])
with col1:
    st.image('app_icon.svg', width=60)
with col2:
    st.title('Bank Customer Churn Predictor')

st.write('Enter customer details to predict the likelihood of churn.')

# Define input fields for all features used in the model
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
gender = st.selectbox('Gender', ['Male', 'Female'], index=0)
age = st.number_input('Age', min_value=18, max_value=100, value=35)
tenure = st.number_input('Tenure (years with bank)', min_value=0, max_value=20, value=5)
balance = st.number_input('Account Balance', min_value=0.0, value=50000.0)
num_of_products = st.selectbox('Number of Products', [1, 2, 3, 4], index=0)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'], index=0)
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'], index=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0)

# Prepare input for prediction
def preprocess_input():
    data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'EstimatedSalary': estimated_salary,
        'Gender_Male': 1 if gender == 'Male' else 0
    }
    return pd.DataFrame([data])

if st.button('Predict Churn'):
    status_container = st.status("Predicting...")
    with status_container as status:
        status_container.update(label="Loading the data into the model...", state="running", expanded=True)
        input_df = preprocess_input()
        st.write("Input DataFrame for prediction:")
        st.dataframe(input_df)
        status_container.update(label="Making the prediction...", state="running", expanded=True)
        prediction = model.predict(input_df)[0]
        status.text("Prediction complete!")
    status_container.update(label="Prediction complete!", state="complete", expanded=False)
    if prediction == 1:
        st.error('This customer is likely to LEAVE.')
    else:
        st.balloons()
        st.success('This customer is likely to STAY!')
