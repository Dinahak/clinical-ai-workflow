import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

def explore():
    train = pd.read_csv(DATA_DIR / "training.csv")
    test  = pd.read_csv(DATA_DIR / "testing.csv")

    train.columns = train.columns.str.strip()
    test.columns  = test.columns.str.strip()

    symptom_cols = [c for c in train.columns if c != "prognosis"]

    print("=" * 55)
    print("TRAINING SET")
    print("=" * 55)
    print(f"  Rows          : {len(train)}")
    print(f"  Symptom cols  : {len(symptom_cols)}")
    print(f"  Diseases      : {train['prognosis'].nunique()}")

    print("\n  All diseases in dataset:")
    diseases = sorted(train["prognosis"].unique())
    counts   = train["prognosis"].value_counts()
    for i, d in enumerate(diseases, 1):
        print(f"    {i:2}. {d:45} {counts[d]} samples")

    print("\n  First 10 symptom columns:")
    for s in symptom_cols[:10]:
        pct = train[s].mean() * 100
        print(f"    - {s:40} present in {pct:.1f}% of cases")

    print("\n=" * 55)
    print("TESTING SET")
    print("=" * 55)
    print(f"  Rows     : {len(test)}")
    print(f"  Diseases : {test['prognosis'].nunique()}")

    print("\n=" * 55)
    print("DATA QUALITY CHECKS")
    print("=" * 55)
    missing_train = train.isnull().sum().sum()
    missing_test  = test.isnull().sum().sum()
    print(f"  Missing values in training : {missing_train}")
    print(f"  Missing values in testing  : {missing_test}")
    print(f"  Value range (should be 0/1): {train[symptom_cols].values.min()} to {train[symptom_cols].values.max()}")

    summary = {
        "n_train"         : len(train),
        "n_test"          : len(test),
        "n_symptoms"      : len(symptom_cols),
        "n_diseases"      : len(diseases),
        "diseases"        : diseases,
        "symptom_columns" : symptom_cols
    }
    out = DATA_DIR / "dataset_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved -> {out}")
    print("\n  NEXT STEP: python data/disease_prediction/processor.py")
    return train, test, symptom_cols, diseases

if __name__ == "__main__":
    explore()
