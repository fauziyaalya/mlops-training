"""
MODUL 4 - LAB SESI 2 - Langkah 1: Multi-Eksperimen
Jalankan 3x dengan kombinasi parameter berbeda, bandingkan AUC di MLflow UI:
    python train.py 100 5
    python train.py 200 8
    python train.py 300 12
Lalu: mlflow ui  -> buka experiment "training-siang", urutkan kolom AUC.
"""
import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print("Cara pakai: python train.py <n_estimators> <max_depth>")
        sys.exit(1)

    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])

    df = pd.read_csv("data/train.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"],
        test_size=0.2, random_state=42)

    mlflow.set_experiment("training-siang")
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        ).fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("auc", auc)
        # serialization_format="pickle" -> hindari error "untrusted types" dari
        # format skops default pada beberapa versi mlflow/scikit-learn terbaru
        mlflow.sklearn.log_model(
            model, "model", serialization_format="pickle"
        )

        print(f"n_estimators={n_estimators} max_depth={max_depth} -> AUC={auc:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
