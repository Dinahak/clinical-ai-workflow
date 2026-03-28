import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    return df


def inspect_data(df, n=5):
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    print(df.head(n))


def clean_data(df):
    # Example steps; adjust to your dataset
    # 1. Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # 2. Drop rows with all missing
    df = df.dropna(axis=0, how="all")

    # 3. Fill numeric missing values with median
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # 4. Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        if not df[c].mode().empty:
            df[c] = df[c].fillna(df[c].mode()[0])

    return df


def encode_data(df):
    # One-hot encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def split_data(df, target_column, test_size=0.2, random_state=42):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )
    return X_train, X_test, y_train, y_test


def save_split_data(X_train, X_test, y_train, y_test, dir_path="data/processed"):

    os.makedirs(dir_path, exist_ok=True)
    X_train.to_csv(f"{dir_path}/X_train.csv", index=False)
    X_test.to_csv(f"{dir_path}/X_test.csv", index=False)
    y_train.to_csv(f"{dir_path}/y_train.csv", index=False)
    y_test.to_csv(f"{dir_path}/y_test.csv", index=False)


def main():
    df = load_data("data/train.csv")
    inspect_data(df)

    df = clean_data(df)
    df = encode_data(df)

    # Replace with your actual target label
    target_column = "target"
    if target_column not in df.columns:
        raise ValueError(f"Expected target column '{target_column}' missing after preprocessing")

    X_train, X_test, y_train, y_test = split_data(df, target_column)
    save_split_data(X_train, X_test, y_train, y_test)

    print("Processed data saved to data/processed")


if __name__ == "__main__":
    main()