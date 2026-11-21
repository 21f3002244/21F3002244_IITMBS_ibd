import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from src.preprocess import (
    FEATURES,
    load_data,
    impute_classwise_last10
)

def main():
    data = load_data("data/iris_v0.csv")

    train_df, val_df = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data["species"]
    )

    train_df = impute_classwise_last10(train_df, train_df)
    val_df = impute_classwise_last10(train_df, val_df)

    model = joblib.load("models/best_model.joblib")
    predictions = model.predict(val_df[FEATURES])

    accuracy = accuracy_score(val_df["species"], predictions)
    report = classification_report(
        val_df["species"],
        predictions,
        output_dict=True,
        zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "classification_report": report
    }

    with open("outputs/ci_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("CI evaluation accuracy:", round(accuracy, 4))
    print(classification_report(val_df["species"], predictions))

if __name__ == "__main__":
    main()
