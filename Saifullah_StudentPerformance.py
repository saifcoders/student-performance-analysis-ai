"""
Student Performance Analysis and Prediction Using AI
Dataset: synthetic illustrative data generated for this educational project.
Run: python saifullah_student_performance.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_performance_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_demo_dataset(rows=600, seed=42):
    """Create reproducible illustrative data; it is not real student data."""
    import numpy as np
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(2.8, 1.2, rows), 0, 7)
    attendance = np.clip(rng.normal(82, 11, rows), 40, 100)
    previous = np.clip(rng.normal(68, 14, rows), 20, 100)
    assignments = np.clip(np.rint(rng.normal(7, 2, rows)), 0, 10).astype(int)
    sleep = np.clip(rng.normal(7, 1.1, rows), 3, 10)
    noise = rng.normal(0, 8, rows)
    score = np.clip(18 + study*5.2 + attendance*.22 + previous*.28
                    + assignments*1.1 + (sleep-6.5)*1.4 + noise, 0, 100)
    category = np.where(score >= 75, "High",
                        np.where(score >= 55, "Medium", "Needs Support"))
    return pd.DataFrame({
        "study_hours_per_day": np.round(study, 1),
        "attendance_percent": np.round(attendance, 1),
        "previous_score": np.round(previous, 1),
        "assignments_completed_out_of_10": assignments,
        "sleep_hours_per_day": np.round(sleep, 1),
        "final_score": np.round(score, 1),
        "performance_category": category
    })

def main():
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = create_demo_dataset()
        data.to_csv(DATA_PATH, index=False)

    print("\\nSTUDENT PERFORMANCE ANALYSIS")
    print("=" * 38)
    print(f"Rows: {len(data)}")
    print("\\nFirst five rows:")
    print(data.head().to_string(index=False))
    print("\\nSummary statistics:")
    print(data.select_dtypes(include="number").describe().round(2).to_string())
    print("\\nPerformance category counts:")
    print(data["performance_category"].value_counts().to_string())

    # Save a category chart
    data["performance_category"].value_counts().reindex(
        ["High", "Medium", "Needs Support"], fill_value=0
    ).plot(kind="bar", title="Student Performance Category Counts")
    plt.xlabel("Performance category")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_categories.png", dpi=150)
    plt.close()

    # Relationship chart
    data.plot.scatter(x="study_hours_per_day", y="final_score",
                      title="Study Hours vs Final Score", alpha=0.65)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "study_hours_vs_score.png", dpi=150)
    plt.close()

    features = [
        "study_hours_per_day", "attendance_percent", "previous_score",
        "assignments_completed_out_of_10", "sleep_hours_per_day"
    ]
    X = data[features]
    y = data["performance_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42,
                                   class_weight="balanced")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\\nRandom Forest test accuracy: {accuracy:.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["High", "Medium", "Needs Support"],
        xticks_rotation=20
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    importance.plot(kind="barh", title="Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # Example prediction: values are hypothetical, not a judgement about a real student.
    example = pd.DataFrame([{
        "study_hours_per_day": 3.0,
        "attendance_percent": 85.0,
        "previous_score": 70.0,
        "assignments_completed_out_of_10": 7,
        "sleep_hours_per_day": 7.0
    }])
    print("\\nExample prediction (hypothetical inputs):",
          model.predict(example)[0])
    print(f"\\nCharts saved in: {OUTPUT_DIR}")
    print("Note: this project uses synthetic demo data; do not use its predictions for real decisions.")

if __name__ == "__main__":
    main()
