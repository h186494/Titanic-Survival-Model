
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

df = pd.read_csv("data/train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X = df[features].copy()
y = df["Survived"]

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

age_median = X_train["Age"].median()
fare_median = X_train["Fare"].median()

X_train = X_train.fillna({
    "Age": age_median,
    "Fare": fare_median
})

X_val = X_val.fillna({
    "Age": age_median,
    "Fare": fare_median
})

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_val)


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0)
    }


baseline_results = evaluate(y_val, baseline_predictions)
model_results = evaluate(y_val, predictions)

print("\nBaseline results:")
for metric, value in baseline_results.items():
    print(f"{metric}: {value:.2%}")

print("\nLogistic Regression results:")
for metric, value in model_results.items():
    print(f"{metric}: {value:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

coefficients["Absolute value"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(
    by="Absolute value", ascending=False
)

print("\nFeature coefficients:")
print(coefficients.to_string(index=False))

joblib.dump(model, "titanic_model.pkl")
print("\nModellen er lagret!")
