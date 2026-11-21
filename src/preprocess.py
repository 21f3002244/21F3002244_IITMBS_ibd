import pandas as pd

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABELS = {"setosa": 0, "versicolor": 1, "virginica": 2,
          "Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

def load_data(path):
    df = pd.read_csv(path)
    df["species"] = df["species"].astype(str).str.strip()
    df["species"] = df["species"].map(LABELS)
    if df["species"].isna().any():
        raise ValueError("Unknown species label found")
    df["species"] = df["species"].astype(int)
    return df

def impute_classwise_last10(train_df, target_df):
    result = target_df.copy()
    for cls in result["species"].unique():
        reference = train_df[train_df["species"] == cls]
        for feature in FEATURES:
            valid = reference[feature].dropna().tail(10)
            fallback = valid.mean() if not valid.empty else train_df[feature].mean()
            mask = (result["species"] == cls) & result[feature].isna()
            result.loc[mask, feature] = fallback
    return result
