# ============================================
# Titanic Survival Prediction Project
# ============================================

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================
# Step 2: Load Dataset
# ============================================

df = pd.read_csv("train.csv")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# ============================================
# Step 3: Data Visualization
# ============================================

plt.figure(figsize=(5,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# ============================================
# Step 4: Data Cleaning
# ============================================

df["Age"].fillna(df["Age"].median(), inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df.drop("Cabin", axis=1, inplace=True)

# ============================================
# Step 5: Feature Engineering
# ============================================

label = LabelEncoder()

df["Sex"] = label.fit_transform(df["Sex"])

df["Embarked"] = label.fit_transform(df["Embarked"])

# ============================================
# Step 6: Select Features
# ============================================

X = df[["Pclass",
        "Sex",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "Embarked"]]

y = df["Survived"]

# ============================================
# Step 7: Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Step 8: Train Model
# ============================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ============================================
# Step 9: Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Step 10: Accuracy
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# ============================================
# Step 11: Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# ============================================
# Step 12: Classification Report
# ============================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ============================================
# Step 13: Predict New Passenger
# ============================================

print("\nPredicting New Passenger")

# Pclass, Sex(0=Female,1=Male), Age, Fare, SibSp, Parch, Embarked
new_passenger = [[3, 1, 25, 7.25, 0, 0, 2]]

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Prediction: Passenger Survived")
else:
    print("Prediction: Passenger Did Not Survive")

# ============================================
# Step 14: Feature Importance
# ============================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nFeature Importance")
print(importance.sort_values(by="Coefficient", ascending=False))

# ============================================
# Step 15: Save Predictions
# ============================================

results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

results.to_csv("Titanic_Predictions.csv", index=False)

print("\nPredictions saved as Titanic_Predictions.csv")

print("\nProject Completed Successfully!")

