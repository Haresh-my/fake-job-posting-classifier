import streamlit as st
import pandas as pd
import cloudpickle as cp
import joblib

# ---- 1. Load saved classifier and preprocessor ----
@st.cache_resource
def load_model():
    with open("fake_job_model.pkl", "rb") as f:
        return cp.load(f)

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

classifier = load_model()
preprocessor = load_preprocessor()

# ---- 2. Scam score function ----
def scam_score(text):
    scam_keywords = [
        "urgent", "click here", "no experience", "work from home", "fast money",
        "instant join", "easy job", "limited slots", "start immediately", "quick cash",
        "daily payout", "paytm", "send your details", "earn money", "apply fast",
        "join now", "registration fee", "no resume", "just apply", "whatsapp us",
        "aadhar card", "online part time", "flexible hours", "no documents required",
        "100% guaranteed", "free registration"
    ]
    return sum(word in text.lower() for word in scam_keywords)

# ---- 3. Streamlit UI ----
st.title("🕵️‍♂️ Fake Job Posting Detector")

st.subheader("Enter the job posting details:")
desc = st.text_area("Description", height=200)
profile = st.text_area("Company Profile", height=200)
salary = st.text_input("Salary Range")
location = st.text_input("Location")
has_logo = st.selectbox("Does the post have a company logo?", ["Yes", "No"])
logo = 1 if has_logo == "Yes" else 0

# ---- 4. Predict Button ----
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'description': desc,
        'company_profile': profile,
        'salary_range': salary,
        'location': location,
        'has_company_logo': logo
    }])

    input_data['scam_score'] = input_data['description'].apply(scam_score)

    try:
        X_transformed = preprocessor.transform(input_data)
        prediction = classifier.predict(X_transformed)

        if prediction[0] == 1:
            st.error("🚩 This job posting is likely **FAKE**!")
        else:
            st.success("✅ This job posting is likely **REAL**.")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
    