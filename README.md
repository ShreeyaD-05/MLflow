# MLflow Experiment Tracking and Data Drift Analysis

## Objective

This project demonstrates how to use MLflow to track machine learning experiments and how to detect data drift in a dataset using statistical testing. It covers training classification models, logging parameters and metrics, simulating data drift, and visualising everything through the MLflow UI.

---

## Technologies Used

| Tool | Version |
|---|---|
| Python | 3.11.9 |
| MLflow | 3.16.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |

---

## Dataset

- **Name:** Iris Dataset
- **Source:** `sklearn.datasets.load_iris`
- **Records:** 150 samples
- **Features:** sepal length, sepal width, petal length, petal width (all in cm)
- **Target:** Species — Setosa, Versicolor, Virginica (3 classes)

---

## Project Structure

```
MLFLOW_LAB/
│
├── mlflow_exp.py          # MLflow classification experiment (RF_default, RF_200trees)
├── drift_analysis.py      # Data drift detection using KS test + MLflow tracking
├── requirements.txt       # Python dependencies
├── mlflow.db              # SQLite tracking database (shared by both scripts)
├── mlruns/                # MLflow run artifacts
│   ├── 1/                 # Iris_Classification_Experiment artifacts
│   └── 2/                 # Iris_Data_Drift_Analysis artifacts
├── .gitignore
└── README.md
```

---

## MLflow Experiment Tracking

**File:** `mlflow_exp.py`

Trains two Random Forest classifiers on the Iris dataset and logs everything to MLflow.

| Run Name | n_estimators | max_depth | Accuracy | F1 Score |
|---|---|---|---|---|
| RF_default | 100 | None | logged | logged |
| RF_200trees | 200 | 5 | logged | logged |

Logged per run:
- **Parameters:** `n_estimators`, `max_depth`, `random_state`, `test_size`
- **Metrics:** `accuracy`, `f1_score`
- **Artifacts:** trained model (`.skops` format)

---

## Data Drift Analysis

**File:** `drift_analysis.py`

### What is Data Drift?

Data drift happens when the statistical distribution of input features in production data shifts away from the training data distribution. This can silently degrade model performance.

### Drift Simulation

To simulate drift, the **petal length** feature is multiplied by **1.30** (a 30% increase) in the current dataset. This is intentional and artificial — it represents what could happen if, for example, a sensor started reporting inflated values in production.

The other three features (sepal length, sepal width, petal width) are kept identical to the reference data, so they should show **no drift**.

### Detection Method: Kolmogorov-Smirnov Test

The **KS two-sample test** (`scipy.stats.ks_2samp`) compares the full distribution of a feature in the reference dataset against the current dataset.

- **KS Statistic:** Maximum distance between the two cumulative distribution functions. Higher = more different.
- **p-value:** Probability of seeing this difference by chance if the distributions were the same.
- **Threshold:** `0.05` — if `p-value < 0.05`, drift is flagged as statistically significant.

The test is run independently on all 4 features.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ShreeyaD-05/MLflow.git
cd MLflow

# Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Code

### Step 1 — Run the classification experiment

```bash
python mlflow_exp.py
```

This trains RF_default and RF_200trees and logs results to `mlflow.db`.

### Step 2 — Run the drift analysis

```bash
python drift_analysis.py
```

This runs the KS test on all 4 features, saves `outputs/drift_results.csv`, and logs a new MLflow run under `Iris_Data_Drift_Analysis`.

### Step 3 — View results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser at: **http://127.0.0.1:5000**

You will see two experiments:
- `Iris_Classification_Experiment` — with runs RF_default and RF_200trees
- `Iris_Data_Drift_Analysis` — with run KS_drift_detection

---

## Expected Output — drift_analysis.py

```
==============================================================
  Data Drift Detection - Kolmogorov-Smirnov Test
  Simulated drift: petal length increased by 30%
==============================================================
Feature                      KS Stat      p-value     Drift?
--------------------------------------------------------------
sepal length (cm)             0.0000     1.000000   No Drift
sepal width (cm)              0.0000     1.000000   No Drift
petal length (cm)             0.3933     0.000000 YES - Drift
petal width (cm)              0.0000     1.000000   No Drift
--------------------------------------------------------------

Total features tested       : 4
Features with drift detected: 1
Significance threshold used : 0.05
==============================================================
```

---

## Results Interpretation

| Feature | KS Statistic | p-value | Drift Detected |
|---|---|---|---|
| sepal length (cm) | 0.0000 | 1.000000 | No |
| sepal width (cm) | 0.0000 | 1.000000 | No |
| petal length (cm) | 0.3933 | ~0.000000 | **Yes** |
| petal width (cm) | 0.0000 | 1.000000 | No |

- **Petal length** shows a very high KS statistic and a p-value effectively zero — confirming the 30% shift is detected with high confidence.
- The other three features have KS stat = 0 and p-value = 1.0, confirming they are identical in both datasets (no drift).
- This validates that the KS test correctly isolates only the modified feature.

---

## Limitations of Simulated Drift

- The drift here is manually injected, not observed from real production data.
- A uniform 30% multiplication creates a very obvious shift — real drift is usually more subtle.
- The KS test detects distributional differences but does not explain the cause.
- In production, drift monitoring would require continuous data pipelines, not a static comparison.

---

## GitHub Repository

[https://github.com/ShreeyaD-05/MLflow](https://github.com/ShreeyaD-05/MLflow)
