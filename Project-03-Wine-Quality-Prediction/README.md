# 🍷 Project: Wine Quality Prediction

## 🎯 Objective:
To build a machine learning model that predicts wine quality based on various physicochemical properties. The project involves cleaning the dataset, performing exploratory data analysis (EDA), training a classifier, and drawing useful insights.

---

## 🧰 Tools & Libraries Used:
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Google Colab

---

## 📦 Dataset:
- *Source:* [WineQuality_Red.csv](https://raw.githubusercontent.com/selva86/datasets/master/WineQuality_Red.csv)
- *Target column:* quality (score from 0–10)
- *Features:* alcohol, sulphates, citric acid, density, etc.

---

## 🔁 Workflow Summary:
1. Imported required libraries
2. Loaded and explored the dataset
3. Removed duplicates and handled class imbalance
4. Converted quality into binary categories (good/bad)
5. Scaled features for better model performance
6. Trained a RandomForestClassifier
7. Evaluated using accuracy, confusion matrix, and classification report
8. Visualized feature correlations and distributions

---

## 🧪 Full Python Code:

python
# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/WineQuality_Red.csv"
df = pd.read_csv(url)

# 3. Explore Dataset
print(df.head())
print(df.info())
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# 4. Clean Data
df.drop_duplicates(inplace=True)
df['quality'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)  # Binary classification

# 5. Feature Scaling
X = df.drop('quality', axis=1)
y = df['quality']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


---

## 📊 Visualizations:

python
# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Wine Features")
plt.show()

# Quality Distribution
sns.countplot(x='quality', data=df)
plt.title("Wine Quality (0 = Bad, 1 = Good)")
plt.show()

# Alcohol vs Quality
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol Content by Wine Quality")
plt.show()


---

## 🔍 Key Insights:
- *Alcohol, **sulphates, and **citric acid* show positive correlation with wine quality.
- *Volatile acidity* and *density* are negatively correlated with quality.
- The majority of wines are of *average or low quality*; few are rated high.
- The binary classification simplifies prediction and improves accuracy.
- The *Random Forest model* achieved high performance in predicting wine quality categories.

---

## ✅ Conclusion:
This project successfully demonstrates how to clean a dataset, convert it into a binary classification problem, apply feature scaling, train a machine learning model, and visualize feature impact. It’s a great starting point for modeling real-world, chemistry-based classification problems.

---

## 📝 Author:
Archi Dwivedi

🚀 Project completed as part of a Data Analytics Internship.
