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
from scipy.sparse import hstack
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data():
    file_path = r"C:\Users\Harsh Sharma\Downloads\Fake_job_postings\fake_job_postings.csv"
    
    try:
        df = pd.read_csv(file_path)
        print(" CSV file loaded successfully.")
        return df
    except FileNotFoundError:
        print(" File not found. Check your path again.")
        return None
# def load_data(zip_path, inner_csv):
#     print("Loading Data from Zip File...")
#     with zipfile.ZipFile(zip_path) as zip_file:
#         with zip_file.open(inner_csv) as f:
#             df = pd.read_csv(f)
#     print(f"✅ Data loaded from zip. Shape: {df.shape}")
#     return df


def clear_missing (df):
    print("Cleaing data from missing values...")

    df.fillna("", inplace=True)
    print(f"Missing values cleared. Shape: {df.shape}")
    return df


def split_data(df):
    print("Splitting data into train and test sets...")
    
    X = df[['description', 'company_profile', 'salary_range', 'location', 'has_company_logo' ,  'scam_score']]
    y = df["fraudulent"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

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


def build_pipeline():
    print("Building the pipeline...")
    preprocessor = ColumnTransformer([
        ('desc_tfidf', TfidfVectorizer(max_features=100), 'description'),
        ('profile_tfidf', TfidfVectorizer(max_features=100), 'company_profile'),
        ('salary_tfidf', TfidfVectorizer(max_features=50), 'salary_range'),
        ('location_tfidf', TfidfVectorizer(max_features=50), 'location'),
        ('logo_enc', OrdinalEncoder(), ['has_company_logo']),
        ('scam_scaler',StandardScaler() , ['scam_score'])

    ])

    my_pipeline = Pipeline([
        ('preprocessor' , preprocessor),
        ('classifier', XGBClassifier( scale_pos_weight=17, use_label_encoder=False, eval_metric='logloss' , random_state=42))
    ])

    print("Pipeline built successfully.")
    return my_pipeline

def new_pipeline():
    print("Building a New pipeline...")
    X_train_transform = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transform, y_train)
    return X_train_resampled, y_train_resampled


def train_model(model, X_train, y_train):
    print("Training the model...")
    model = pipeline.named_steps['classifier']
    model.fit(X_train_resampled, y_train_resampled)
    print("Model trained successfully.")
    return model

def save_model(model, path="fake_job_model.pkl"):
    with open(path, "wb") as f:
        cp.dump(model, f)
    print(f"✅ Model saved successfully as {path}")


def evaluate_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print("Model evaluation completed.")

def print_classification_report(y_test, y_pred):
    print("Generating classification report...")
    report = classification_report(y_test, y_pred, target_names=["Not Fraudulent", "Fraudulent"])
    print(report)
    return report


def main():
    df = load_data()
    # df = load_data("C:/Users/Harsh Sharma/Downloads/Fake_job_postings.zip", "fake_job_postings.csv")
    df = clear_missing(df)
    df['scam_score'] = df['description'].apply(scam_score)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline()

    # Preprocess training data
    X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    preprocessor = pipeline.named_steps['preprocessor']
    joblib.dump(preprocessor, 'preprocessor.pkl')


    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    # Train classifier
    classifier = pipeline.named_steps['classifier']
    classifier.fit(X_train_resampled, y_train_resampled)

    # Transform and predict on test set
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    y_pred = classifier.predict(X_test_transformed)

    print_classification_report(y_test, y_pred)
    evaluate_model(classifier, X_test_transformed, y_test)
    save_model(classifier)
    #prediction and evaluation
    y_pred = classifier.predict(X_test)
    print_classification_report(y_test, y_pred)
    
# ✅ Run Everything
if __name__ == "__main__":
    main()