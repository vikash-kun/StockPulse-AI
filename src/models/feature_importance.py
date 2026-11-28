import pandas as pd
import joblib

from pathlib import Path
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "decision_tree.joblib"

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def analyze_feature_importance():

    print("Loading Decision Tree model...")

    model = joblib.load(MODEL_FILE)

    print("Loading training dataset...")

    df = pd.read_csv(TRAIN_FILE)

    df.columns = df.columns.str.strip()

    # Remove target and date
    X = df.drop(columns=["Target", "Date"], errors="ignore")

    # Extract trained decision tree
    classifier = model

    if hasattr(model, "named_steps"):
        classifier = model.named_steps["classifier"]

    # Calculate feature importance
    importance = classifier.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    # Sort features
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== FEATURE IMPORTANCE ==========\n")

    print(importance_df.to_string(index=False))

    # Save results
    output_file = REPORT_DIR / "feature_importance.csv"

    importance_df.to_csv(output_file, index=False)

    print("\nFeature importance saved:")
    print(output_file)

    # Plot
    plt.figure(figsize=(12, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")

    plt.ylabel("Features")

    plt.title("Decision Tree Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plot_file = REPORT_DIR / "feature_importance.png"

    plt.savefig(plot_file)

    plt.close()

    print("\nFeature importance chart saved:")
    print(plot_file)


if __name__ == "__main__":

    analyze_feature_importance()