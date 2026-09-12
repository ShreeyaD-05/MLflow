import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Iris_Classification_Experiment")

with mlflow.start_run(run_name="RF_default"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "RandomForest_model")
    print(f"Run 1 - Accuracy: {acc:.4f}  F1: {f1:.4f}")

with mlflow.start_run(run_name="RF_200trees"):
    model2 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_test)
    acc2 = accuracy_score(y_test, preds2)
    f1_2 = f1_score(y_test, preds2, average='weighted')
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc2)
    mlflow.log_metric("f1_score", f1_2)
    mlflow.sklearn.log_model(model2, "RandomForest_200_model")
    print(f"Run 2 - Accuracy: {acc2:.4f}  F1: {f1_2:.4f}")

print("\nDone! Now run:  mlflow ui --backend-store-uri sqlite:///mlflow.db")