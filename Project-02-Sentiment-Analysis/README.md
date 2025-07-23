# 📊 Sentiment Analysis Project

This project involves building a machine learning model to classify tweets into Positive or Negative sentiments. It is designed for beginners and includes step-by-step data cleaning, exploration, model building, evaluation, and observations.

---

## 📌 Objectives

- Understand text data and preprocessing techniques.
- Explore Natural Language Processing (NLP) pipeline.
- Build classification models using Scikit-learn.
- Evaluate model performance using accuracy, confusion matrix, and classification report.

---

## 🧹 Data Preprocessing

**Steps Followed:**

- ✅ Removed URLs, mentions (@username), hashtags (#tag), emojis, and HTML tags.
- ✅ Removed punctuations, numbers, and converted all text to lowercase.
- ✅ Removed stopwords using NLTK.
- ✅ Tokenized the text data and lemmatized it.
- ✅ Used TfidfVectorizer to convert text into numerical features.

---

## 📊 Exploratory Data Analysis (EDA)

**Key Insights:**

- 📌 Created WordClouds for both positive and negative sentiments.
- 📌 Countplot to visualize class distribution showed data is fairly balanced.
- 📌 Top frequent words in each sentiment class gave insight into how people express emotions.

---

## 🤖 Model Building

**Algorithms Used:**

- Logistic Regression
- Naive Bayes

**Techniques:**

- Trained models using TF-IDF vectors.
- Split data using train_test_split.
- Evaluated performance on test set.

---

## 📈 Evaluation

**Metrics Used:**

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

**Findings:**

- Logistic Regression performed better with ~85% accuracy.
- Balanced performance across both classes.

---

## 📝 Key Observations

1. Dataset contains labeled tweets for binary sentiment classification.
2. Cleaned data by removing special characters, links, mentions, and converting to lowercase.
3. Tokenized text and removed stopwords.
4. WordClouds showed clear distinction between positive and negative sentiments.
5. Class distribution was balanced.
6. Used TF-IDF vectorization for feature extraction.
7. Trained models: Naive Bayes and Logistic Regression.
8. Logistic Regression achieved ~85% accuracy.
9. Confusion matrix showed good classification with low false rates.
10. Logistic Regression performed better than Naive Bayes.
11. Can be improved with N-grams or deep learning (LSTM, BERT).
12. Useful for brand monitoring, feedback analysis, social media.
13. Recommended to use larger datasets for better accuracy.
14. Future work: real-time analysis using Twitter API and deployment.

---

## 🔄 Future Improvements

- Use more advanced NLP models like LSTM, BERT.
- Implement N-gram features for deeper language context.
- Connect to Twitter API for live sentiment classification.
- Deploy the model using Streamlit or Flask.
- Add a feature to analyze trending hashtags and compare sentiment in real time.
- Build a web interface for end-users to upload CSV and get real-time results.

---

## 💡 Requirements

Install required libraries using:

```bash
pip install -r requirements.txt
