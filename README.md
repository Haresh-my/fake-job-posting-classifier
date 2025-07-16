         
                                      ## 🌐 Live DemoTry the app here 👉 [fake-job-posting-classifier.onrender.com](https://fake-job-posting-classifier.onrender.com)


# fake-job-posting-classifier 


# 🕵️‍♂️ Fake Job Detector

## [![Live App](https://img.shields.io/badge/Live-Fake_Job_Classifier-green?style=for-the-badge&logo=streamlit)](https://fake-job-posting-classifier.onrender.com)


Detect fake job postings using Machine Learning.  
This project uses **TF-IDF for text feature extraction**, **XGBoost** for classification, and handles **class imbalance** in a real-world job dataset.

---

## 📂 Dataset

- Source: [Kaggle - Real or Fake Job Posting](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Total posts: ~18,000+
- Class distribution:
  - **Real** jobs: ~17,000
  - **Fake** jobs: ~800 (highly imbalanced!)

---

## 💡 Problem Statement

Can we build a model that:
- Learns from job **description, company profile, salary, location, logo**
- Predicts whether a job posting is **fake or real**
- Works even with class imbalance

---

## 🔧 Features Used

| Feature            | Type     | Engineered |
|-------------------|----------|------------|
| `description`      | Text     | ✅ TF-IDF |
| `company_profile`  | Text     | ✅ TF-IDF |
| `salary_range`     | Text     | ✅ TF-IDF |
| `location`         | Text     | ✅ TF-IDF |
| `has_company_logo` | Binary   | ✅ Encoded |

---

## 🤖 Model Pipeline

```python
Pipeline([
    ('preprocessing', ColumnTransformer(...TF-IDF...)),
    ('classifier', XGBClassifier(scale_pos_weight=17))
])

Performance
Metric	Score
Accuracy	98%
Precision (Fake)	78%
Recall (Fake)	76%
F1-Score (Fake)	77%


How to Run
Clone the repo:

bash
Copy
Edit
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector

pip install -r requirements.txt

jupyter notebook Fake_Job_Posting_Detection.ipynb

💡 Improvements
Add sentence embeddings (BERT/SBERT) instead of TF-IDF

Deploy with Streamlit or Flask

Auto alert for suspicious job patterns

🙌 Author
Harsh Sharma
Passionate about building intelligent systems that think, talk, and learn like humans 🧠

📜 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

### ✅ Next Steps

- Save this content into a file called `README.md`
- Place it in your GitHub repo root folder
- Optionally create a `requirements.txt` too:
```txt
xgboost
scikit-learn
pandas
matplotlib
seaborn
wordcloud
jupyter

## Classification Report
<img width="640" height="480" alt="fake job post" src="https://github.com/user-attachments/assets/fbc0077f-a0f4-4387-9cdb-6ae29427416d" />


