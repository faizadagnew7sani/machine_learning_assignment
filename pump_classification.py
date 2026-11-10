import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)


# STEP 1: Load the dataset

data = pd.read_csv("pump_data_240.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

print("\nClass distribution:")
print(data["Condition"].value_counts())


# STEP 2: Select input features and target

X = data[
    [
        "Vibration",
        "RPM",
        "Temperature",
        "Pressure",
        "Operating_Hours"
    ]
]

y = data["Condition"]


# STEP 3: Split the dataset
# 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 4: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# STEP 5: Train the model

model.fit(X_train, y_train)