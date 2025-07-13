import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import cloudpickle as cp

# 1. Load data
df = pd.read_csv("Fake_Job_Posting_Detection.csv")
df.fillna("", inplace=True)

X = df[['description', 'company_profile', 'salary_range', 'location', 'has_company_logo']]
y = df["fraudulent"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build clean pipeline (no custom code/classes)
preprocessor = ColumnTransformer([
    ('desc', TfidfVectorizer(max_features=100), 'description'),
    ('profile', TfidfVectorizer(max_features=100), 'company_profile'),
    ('salary', TfidfVectorizer(max_features=50), 'salary_range'),
    ('location', TfidfVectorizer(max_features=50), 'location'),
    ('logo', OrdinalEncoder(), ['has_company_logo'])
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Save using cloudpickle
with open("fake_job_model.pkl", "wb") as f:
    cp.dump(pipeline, f)

print("✅ Clean model trained and saved successfully.")
