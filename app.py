import streamlit as st
import pandas as pd
import joblib

# Load your saved model
model = joblib.load("fake_job_model.joblib")

# App title
st.title("🕵️‍♂️ Fake Job Posting Detector")

st.subheader("Enter the posting details :")
desc = st.text_area("description", height=200)
profile = st.text_area("company_profile", height=200)
salary = st.text_input("salary_range")
location = st.text_input("location")
has_logo = st.selectbox("has_company_logo", ["Yes", "No"])

logo = 1 if has_logo == "Yes" else 0

if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'description': [desc],
        'company_profile': [profile],
        'salary_range': [salary],
        'location': [location],
        'has_company_logo': [logo]
    })

    # Make prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("This job posting is likely to be **FAKE**! 🚩" )

    else :
        st.success("This job posting is likely to be **REAL**! ✅")