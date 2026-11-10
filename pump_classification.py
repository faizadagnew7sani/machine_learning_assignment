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
# STEP 6: Make predictions

y_pred = model.predict(X_test)


# STEP 7: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nAccuracy:")
print(f"{accuracy:.4f}")


# STEP 8: Calculate Fault Precision and Fault Recall

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    y_pred,
    labels=["Fault"],
    average=None,
    zero_division=0
)

fault_precision = precision[0]
fault_recall = recall[0]

print("\nFault Precision:")
print(f"{fault_precision:.4f}")

print("\nFault Recall:")
print(f"{fault_recall:.4f}")


# STEP 9: Calculate Macro F1-score

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nMacro F1-score:")
print(f"{macro_f1:.4f}")


# STEP 10: Classification Report

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Normal", "Warning", "Fault"],
        zero_division=0
    )
)
# STEP 11: Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Normal", "Warning", "Fault"]
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# STEP 12: Display Confusion Matrix

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Warning", "Fault"]
)

disp.plot()

plt.title("Pump Condition Classification - Random Forest")
plt.tight_layout()
plt.show()


# STEP 13: Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(importance)


# STEP 14: Display Feature Importance

importance.plot(kind="bar")

plt.title("Feature Importance - Random Forest")
plt.xlabel("Pump Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 15: Predict a New Pump Observation

new_pump = pd.DataFrame({
    "Vibration": [6.8],
    "RPM": [1580],
    "Temperature": [92],
    "Pressure": [3.0],
    "Operating_Hours": [2400]
})

prediction = model.predict(new_pump)

print("\n==============================")
print("NEW PUMP PREDICTION")
print("==============================")

print("Prediction:", prediction[0])
