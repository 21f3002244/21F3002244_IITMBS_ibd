import json
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .preprocess import FEATURES, load_data, impute_classwise_last10

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_OPPE")

v0 = load_data("data/iris_v0.csv")
v1 = load_data("data/iris_v1.csv")

# Keep the same held-out validation set for both iterations.
train_v0, val_v0 = train_test_split(
    v0, test_size=0.20, random_state=42, stratify=v0["species"]
)
val_v0 = impute_classwise_last10(train_v0, val_v0)
train_v0 = impute_classwise_last10(train_v0, train_v0)

train_merged = pd.concat([train_v0, v1], ignore_index=True)
train_merged = impute_classwise_last10(train_merged, train_merged)

X_val, y_val = val_v0[FEATURES], val_v0["species"]
trials = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "svc": SVC(probability=True, random_state=42)
}

best_score = -1
best_model = None
best_info = {}

for iteration, train_df in [(1, train_v0), (2, train_merged)]:
    train_df = impute_classwise_last10(train_df, train_df)
    X_train, y_train = train_df[FEATURES], train_df["species"]

    for name, estimator in trials.items():
        with mlflow.start_run(run_name=f"iteration_{iteration}_{name}"):
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, predictions, average="weighted", zero_division=0
            )

            mlflow.log_param("iteration", iteration)
            mlflow.log_param("model", name)
            mlflow.log_param("training_rows", len(train_df))
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1
            })
            mlflow.sklearn.log_model(
                estimator,
                artifact_path="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            report = classification_report(
                y_val, predictions, output_dict=True, zero_division=0
            )
            report_path = f"outputs/iteration_{iteration}_{name}_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)

            print(f"Iteration {iteration} | {name} | "
                  f"accuracy={accuracy:.4f} | F1={f1:.4f}")

            if accuracy > best_score:
                best_score = accuracy
                best_model = estimator
                best_info = {
                    "iteration": iteration,
                    "model": name,
                    "accuracy": accuracy,
                    "precision_weighted": precision,
                    "recall_weighted": recall,
                    "f1_weighted": f1
                }

joblib.dump(best_model, "models/best_model.joblib")
with open("outputs/best_model_metrics.json", "w") as f:
    json.dump(best_info, f, indent=2)

print("\nBEST MODEL")
print(json.dumps(best_info, indent=2))
print("\nSaved models/best_model.joblib")
