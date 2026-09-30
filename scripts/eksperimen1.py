"""
MODUL 3 - LAB SESI 1 - Langkah 4: Eksperimen Pertama dengan MLflow
Jalankan: python eksperimen1.py
Lalu:     mlflow ui   (buka http://localhost:5000)
"""
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("data/train_v3.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"],
    test_size=0.2, random_state=42)

mlflow.set_experiment("training-pagi")
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    ).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    print(f"Akurasi: {acc:.4f}")
