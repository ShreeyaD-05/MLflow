# Data Drift Detection using Kolmogorov-Smirnov Test
# Dataset  : Iris (sklearn)
# Method   : KS two-sample test (scipy)
# Tracking : MLflow (sqlite:///mlflow.db)

import os
import pandas as pd
import mlflow
from sklearn.datasets import load_iris
from scipy.stats import ks_2samp

# ----------------------------------------------------------
# 1. Load Iris dataset
# ----------------------------------------------------------
iris = load_iris()

# Only the 4 numerical feature columns (ignore target labels)
feature_names = iris.feature_names
# ['sepal length (cm)', 'sepal width (cm)',
#  'petal length (cm)', 'petal width (cm)']

# Reference dataset = original Iris feature values
reference_data = pd.DataFrame(iris.data, columns=feature_names)

# Current dataset = simulated data
# petal length (index 2) is increased by 30% to simulate drift
# all other features remain unchanged
current_data = reference_data.copy()
current_data["petal length (cm)"] = current_data["petal length (cm)"] * 1.30

print("=" * 62)
print("  Data Drift Detection - Kolmogorov-Smirnov Test")
print("  Simulated drift: petal length increased by 30%")
print("=" * 62)
print(f"{'Feature':<26} {'KS Stat':>9} {'p-value':>12} {'Drift?':>10}")
print("-" * 62)

# ----------------------------------------------------------
# 2. Run KS test on each of the 4 features
# ----------------------------------------------------------
SIGNIFICANCE_THRESHOLD = 0.05

results = []

for feature in feature_names:
    ref_col = reference_data[feature].values
    cur_col = current_data[feature].values

    ks_stat, p_value = ks_2samp(ref_col, cur_col)

    drift_detected = p_value < SIGNIFICANCE_THRESHOLD
    drift_label    = "YES - Drift" if drift_detected else "No Drift"

    print(f"{feature:<26} {ks_stat:>9.4f} {p_value:>12.6f} {drift_label:>10}")

    results.append({
        "feature"        : feature,
        "ks_statistic"   : round(ks_stat, 6),
        "p_value"        : round(p_value, 6),
        "drift_detected" : drift_detected,
        "drift_status"   : drift_label
    })

print("-" * 62)

total_features   = len(results)
drifted_features = sum(1 for r in results if r["drift_detected"])

print(f"\nTotal features tested       : {total_features}")
print(f"Features with drift detected: {drifted_features}")
print(f"Significance threshold used : {SIGNIFICANCE_THRESHOLD}")
print("=" * 62)

# ----------------------------------------------------------
# 3. Save results to CSV
# ----------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
csv_path = os.path.join("outputs", "drift_results.csv")

results_df = pd.DataFrame(results)
results_df.to_csv(csv_path, index=False)
print(f"\nDrift results saved to: {csv_path}")

# ----------------------------------------------------------
# 4. Log to MLflow
#    Separate experiment so existing classification runs
#    (RF_default, RF_200trees) are NOT affected at all
# ----------------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Data_Drift_Analysis")

with mlflow.start_run(run_name="KS_drift_detection"):

    # Tags
    mlflow.set_tag("drift_type",        "Simulated covariate drift")
    mlflow.set_tag("drift_description", "Petal length increased by 30% in current dataset")
    mlflow.set_tag("dataset",           "Iris (sklearn)")
    mlflow.set_tag("test_method",       "Kolmogorov-Smirnov two-sample test")

    # Parameters
    mlflow.log_param("significance_threshold",  SIGNIFICANCE_THRESHOLD)
    mlflow.log_param("total_features_tested",   total_features)
    mlflow.log_param("simulated_drift_feature", "petal length (cm)")
    mlflow.log_param("drift_simulation_factor", 1.30)

    # Per-feature metrics
    for r in results:
        key = (r["feature"]
               .replace(" ", "_")
               .replace("(", "")
               .replace(")", "")
               .replace("__", "_"))
        mlflow.log_metric(f"ks_stat_{key}",   r["ks_statistic"])
        mlflow.log_metric(f"p_value_{key}",   r["p_value"])
        mlflow.log_metric(f"drift_flag_{key}", int(r["drift_detected"]))

    # Summary
    mlflow.log_metric("total_drifted_features", drifted_features)

    # Artifact
    mlflow.log_artifact(csv_path, artifact_path="drift_results")

    print("\nMLflow run logged under experiment: Iris_Data_Drift_Analysis")
    print("View UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
