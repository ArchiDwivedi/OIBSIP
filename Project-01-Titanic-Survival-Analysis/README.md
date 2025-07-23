# 🧪 Project: Titanic Dataset – Beginner Data Analytics

## 🎯 Objective:
To analyze and clean the Titanic dataset using Python, perform basic data visualizations, and extract meaningful insights on passenger survival patterns.

## 🧰 Tools & Libraries Used:
- Python
- Pandas
- Seaborn
- Matplotlib

## 📦 Dataset:
- Source: [Titanic Dataset (GitHub)](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- Features: Survived, Pclass, Sex, Age, Fare, Embarked, etc.

## 📋 Steps Performed:
1. Loaded the Titanic dataset directly from GitHub
2. Explored data shape, types, and missing values
3. Cleaned the dataset:
   - Filled missing Age with mean
   - Dropped missing Embarked entries
   - Removed irrelevant columns (Cabin, Ticket, Name)
   - Encoded Sex and Embarked
4. Visualized:
   - Survival by gender
   - Age distribution
   - Correlation heatmap
5. Exported the cleaned dataset to CSV

## 🧠 Final Insights:
- Most survivors were *female*.
- *First class* passengers had higher survival rates.
- The *age range of 20–40* had the most passengers.
- Gender and passenger class had the strongest influence on survival.

## 🧪 Full Python Code:

```python
# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 3: Preview the Data
df.head()

# Step 4: Basic Information
print("Shape of dataset:", df.shape)
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

# Step 5: Clean the Data
df['Age'].fillna(df['Age'].mean(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Step 6: Visualizations
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Save Cleaned Dataset
df.to_csv('cleaned_titanic.csv', index=False)
