import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import zipfile
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle as cp


def load_data(zip_path, inner_csv):
    print("Loading Data from Zip File...")
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open(inner_csv) as f:
            df = pd.read_csv(f)
    print(f"✅ Data loaded from zip. Shape: {df.shape}")
    return df


def clear_missing (df):
    print("Cleaing data from missing values...")
    df.fillna("", inplace=True)
    print(f"Missing values cleared. Shape: {df.shape}")
    return df


def split_data(df):
    print("Splitting data into train and test sets...")
    X = df[['description', 'company_profile', 'salary_range', 'location', 'has_company_logo']]
    y = df["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def build_pipeline():
    print("Building the pipeline...")
    preprocessor = ColumnTransformer([
        ('desc_tfidf', TfidfVectorizer(max_features=100), 'description'),
        ('profile_tfidf', TfidfVectorizer(max_features=100), 'company_profile'),
        ('salary_tfidf', TfidfVectorizer(max_features=50), 'salary_range'),
        ('location_tfidf', TfidfVectorizer(max_features=50), 'location'),
        ('logo_enc', OrdinalEncoder(), ['has_company_logo'])

    ])

    my_pipeline = Pipeline([
        ('preprocessor' , preprocessor),
        ('classifier', XGBClassifier( scale_pos_weight=17, use_label_encoder=False, eval_metric='logloss' , random_state=42))
    ])

    print("Pipeline built successfully.")
    return my_pipeline


    
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


def save_model(model, path="fake_job_model.pkl"):
    with open(path, "wb") as f:
        cp.dump(model, f)
    print(f"✅ Model saved successfully as {path}")


def main():
    df = load_data("C:/Users/Harsh Sharma/Downloads/Fake_job_postings.zip", "fake_job_postings.csv")
    df = clear_missing(df)
    df['scam_score'] = df['description'].apply(scam_score)
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_pipeline()
    model.fit(X_train, y_train)
    save_model(model)

    
# ✅ Run Everything
if __name__ == "__main__":
    main()