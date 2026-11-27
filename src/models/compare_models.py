import pandas as pd
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# Features
FEATURE_COLUMNS = [
    "Daily_Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "MA_5",
    "MA_10",
    "MA_20",
    "Price_MA_5_Ratio",
    "Price_MA_20_Ratio",
    "Volatility_5",
    "Volatility_10",
    "Volatility_20",
    "Volume_Change",
    "Volume_MA_5_Ratio",
    "High_Low_Range",
    "Open_Close_Change"
]


def load_data():

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Target"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Target"]

    return X_train, X_test, y_train, y_test


def compare_models():

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        ),

        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        )
    }

    results = []

    print("\nStarting model comparison...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        balanced_accuracy = balanced_accuracy_score(
            y_test,
            predictions
        )

        probabilities = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, probabilities)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "ROC-AUC": roc_auc
        })

        # Save trained model
        filename = name.lower().replace(" ", "_") + ".joblib"

        joblib.dump(model, MODEL_DIR / filename)

        print(f"{name} completed!")

    # Create comparison table
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Balanced Accuracy",
        ascending=False
    )

    print("\n========== MODEL COMPARISON ==========\n")

    print(results_df.to_string(index=False))

    # Save results
    output_file = REPORT_DIR / "model_comparison.csv"

    results_df.to_csv(output_file, index=False)

    print("\nComparison report saved:")
    print(output_file)


if __name__ == "__main__":
    compare_models()