from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODEL_DIR / "logistic_regression.joblib"


# Features used for prediction
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

TARGET_COLUMN = "Target"


def load_datasets():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}"
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    # Clean column names
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    return train_df, test_df


def prepare_data(train_df, test_df):

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    for name, df in [
        ("Training", train_df),
        ("Testing", test_df)
    ]:
        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset missing columns: {missing}"
            )

        if df[required_columns].isnull().any().any():
            raise ValueError(
                f"{name} dataset contains missing values."
            )

        if not df[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(
                f"{name} target must contain only 0 and 1."
            )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    # Scaling and model training in one pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression model...")

    model.fit(X_train, y_train)

    print("Model training completed!")

    return model


def evaluate_model(model, X_test, y_test):

    print("\nMaking predictions...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n========== MODEL EVALUATION ==========")

    print(
        f"\nAccuracy: "
        f"{accuracy_score(y_test, predictions):.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )

    if y_test.nunique() == 2:
        print(
            f"ROC-AUC: "
            f"{roc_auc_score(y_test, probabilities):.4f}"
        )
    else:
        print("ROC-AUC cannot be calculated: test set has one class.")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "No Increase (0)",
                "Increase (1)"
            ],
            zero_division=0
        )
    )


def save_model(model):

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully!")
    print(f"Location: {MODEL_FILE}")


def main():

    print("Loading training and testing datasets...")

    train_df, test_df = load_datasets()

    print(f"Training records: {len(train_df)}")
    print(f"Testing records: {len(test_df)}")

    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df
    )

    print(f"\nNumber of features: {len(FEATURE_COLUMNS)}")

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()