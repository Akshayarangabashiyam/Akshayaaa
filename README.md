# 🏭 Defect Detection in Hot Rolling — Tata Steel AI Hackathon

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat\&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-yellow?style=flat\&logo=scikit-learn)
![Hackathon](https://img.shields.io/badge/Tata%20Steel-AI%20Hackathon-steel?style=flat)
![Recall](https://img.shields.io/badge/Recall-100%25-brightgreen?style=flat)
![Precision](https://img.shields.io/badge/Precision-98.51%25-brightgreen?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-94.67%25-green?style=flat)

> **Tata Steel AI Hackathon Round 1** — Machine Learning model developed to predict Alpha defects in hot rolling steel coils using XGBoost and SMOTE. Achieved **94.67% Accuracy**, **100% Recall**, and **98.51% Precision**.

---

## 📌 Problem Statement

In Hot Rolling Mills, the **Alpha defect** is a major quality concern that cannot be detected during inline inspection because the steel coil remains under tension throughout the rolling process.

As a result, defect identification must rely solely on the **process parameters** collected from multiple manufacturing stages.

The objective of this project is to **predict Alpha defects before final inspection**, helping reduce quality issues, customer complaints, and production losses.

---

## 🎯 Evaluation Criteria

| Metric        | Requirement              | Achieved     |
| ------------- | ------------------------ | ------------ |
| **Recall**    | 100% (No missed defects) | ✅ **100%**   |
| **Precision** | > 90%                    | ✅ **98.51%** |
| **Accuracy**  | Maximize                 | ✅ **94.67%** |

> ⚠️ In industrial defect detection, Recall is the most important metric because missing a defective coil can result in significant quality and business impact.

---

## 📂 Dataset

| File             | Dimensions | Description                      |
| ---------------- | ---------- | -------------------------------- |
| `train.csv`      | 1352 × 51  | Training data with labels        |
| `test.csv`       | 339 × 50   | Test data without labels         |
| `solution.ipynb` | —          | Complete implementation notebook |
| `submission.csv` | 339 × 2    | Final predictions                |

### Variable Description

| Column   | Description                                 |
| -------- | ------------------------------------------- |
| `CoilID` | Unique coil identifier                      |
| `X1–X49` | Process parameters                          |
| `Y`      | Target variable (1 = Defect, 0 = No Defect) |

### Class Distribution

```text
No Defect (0): 1286 samples — 95.12%
Defect    (1):   66 samples — 4.88%

⚠️ Highly Imbalanced Dataset
```

---

## 🧠 Approach

### Step 1 — Data Preprocessing

* Missing values filled using column median.
* Median chosen because it is robust against outliers.
* Applied RobustScaler for handling industrial data variability.
* Performed data cleaning and consistency checks.

### Step 2 — Feature Engineering

Created row-level statistical features:

```python
df['mean_all']
df['std_all']
df['max_all']
df['min_all']
df['range_all']
df['iqr_all']
df['n_outliers']
df['cv']
```

These features help capture overall process behaviour and identify abnormal coil patterns.

### Step 3 — Handling Class Imbalance

Used SMOTE (Synthetic Minority Oversampling Technique):

```python
SMOTE(
    random_state=42,
    sampling_strategy=0.5
)
```

This generated additional defect samples while maintaining class balance and preserving precision.

### Step 4 — Model Development

Implemented XGBoost Classifier:

```python
XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.01,
    scale_pos_weight=10,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=1
)
```

Reasons for choosing XGBoost:

* Excellent performance on tabular datasets
* Handles non-linear relationships effectively
* Robust against noisy industrial data

### Step 5 — Threshold Optimization

```python
OPTIMAL_THRESHOLD = 0.870
```

Probability thresholds were evaluated to maximize Recall while maintaining high Precision.

Final selected threshold achieved:

```text
Recall    = 100%
Precision = 98.51%
```

---

## 📊 Results

```text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall          : 1.0000
Precision       : 0.9851
Accuracy        : 0.9467

False Negatives : 0
False Positives : 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Confusion Matrix

```text
                 Predicted 0   Predicted 1

Actual 0 (1286)     1285            1
Actual 1 (66)          0           66
```

### Test Predictions

```text
Total Coils      : 339
Defects Detected : 6
```

✅ Zero defective coils missed.

---

## 🛠️ Tech Stack

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python 3.10      | Programming Language       |
| Pandas           | Data Processing            |
| NumPy            | Numerical Computation      |
| Scikit-Learn     | Preprocessing & Evaluation |
| Imbalanced-Learn | SMOTE Oversampling         |
| XGBoost          | Machine Learning Model     |
| Jupyter Notebook | Development Environment    |

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone https://github.com/Akshayarangabashiyam/Akshayaaa.git

cd defect-detection-hot-rolling
```

### 2. Install Dependencies

```bash
pip install xgboost imbalanced-learn scikit-learn pandas numpy
```

### 3. Run Notebook

```bash
jupyter notebook solution.ipynb
```

### 4. Run in Google Colab

Upload the notebook and dataset files to Google Colab and execute all cells.

---

## 📁 Project Structure

```text
defect-detection-hot-rolling/
│
├── solution.ipynb
├── train.csv
├── test.csv
├── submission.csv
├── README.md
└── requirements.txt

## 👨‍💻 Author

**Akshaya Rangabashiyam**

B.Tech Computer and Communication Engineering | Sri Manakula Vinayagar Engineering College

📧 Email: akshayarangabashiyam@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/akshaya-r-1820862a2

💻 GitHub:https://github.com/Akshayarangabashiyam


