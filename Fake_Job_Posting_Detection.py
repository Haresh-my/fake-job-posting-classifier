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

def data(zip_path ,inner_csv ):
    print("Loading Data from Zip File...")
    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open("fake_job_postings.csv") as f:
            df = pd.read_csv(f)

    print(f" Data loaded from zip. Shape: {df.shape}")
    return df


def clear_missing (df):
    print("Cleaing data from missing values...")
    df.fillna("", inplace=True)
    print(f"Missing values cleared. Shape: {df.shape}")
    return df


def spliting_data(df):
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

def train_model(model, X_test, y_test):
    print("Evaluation of the Model...")
    y_pred = model.predict(X_test)
    print("Model evaluation completed.")
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print( "\n Classification report : \n " , classification_report(y_test, y_pred))
    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.show()
    # print("Model training completed.")

def save_model(model, path = "fake_job_model.joblib"):
    print(f"Saving model to {path}...")
    joblib.dump(model, path)
    print("Model saved successfully.")


def main():
    try:
        df = data(r"C:\Users\Harsh Sharma\Downloads\fake_job_postings.zip", "fake_job_postings.csv")
        df = clear_missing(df)
        X_train, X_test, y_train, y_test = spliting_data(df)
        
        
        model = build_pipeline()
        print(" Training model...")
        
        model.fit(X_train, y_train)
        print(" Model trained successfully.")
        
        
        train_model(model, X_test, y_test)
        save_model(model)

    except Exception as e:
        print(f"An error occurred: {e}")
    
# ✅ Run Everything
if __name__ == "__main__":
    main()