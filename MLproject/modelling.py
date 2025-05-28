import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil path relatif ke MLproject/
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Direktori MLproject/
    file_path = sys.argv[3] if len(sys.argv) > 3 else "train_pca.csv"
    # Resolve path relatif ke MLproject/
    file_path = os.path.join(base_dir, file_path)

    # Debug: Print file path
    print(f"Attempting to read file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        # Hapus duplikat fit (ga perlu)
        # model.fit(X_train, y_train)
        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
