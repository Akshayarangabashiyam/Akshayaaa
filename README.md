# 🏭 Defect Detection in Hot Rolling — Tata Steel AI Hackathon

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-yellow?style=flat&logo=scikit-learn)
![Hackathon](https://img.shields.io/badge/Tata%20Steel-AI%20Hackathon-steel?style=flat)
![Recall](https://img.shields.io/badge/Recall-100%25-brightgreen?style=flat)
![Precision](https://img.shields.io/badge/Precision-98.51%25-brightgreen?style=flat)
![Accuracy](https://img.shields.io/badge/Accuracy-94.67%25-green?style=flat)

> **Tata Steel AI Hackathon Round 1** — ML model to predict Alpha defects in hot rolling steel coils using XGBoost and SMOTE. Achieved **94.67% accuracy** with **100% Recall** and **98.51% Precision**.

---

## 📌 Problem Statement

In Hot Rolling Mills, the **Alpha defect** is a critical quality issue that cannot be detected inline because the coil remains under tension during inspection. This means defect prediction must happen using only the **process parameters** recorded across multiple production stages.

The goal is to **detect Alpha defects early** to prevent customer complaints and reduce product downgrades.

---

## 🎯 Evaluation Criteria

| Metric | Requirement | Achieved |
|--------|------------|---------|
| **Recall** | 100% (Zero false negatives) | ✅ **100%** |
| **Precision** | > 90% | ✅ **98.51%** |
| **Accuracy** | Maximize | ✅ **94.67%** |

> ⚠️ **Why Recall matters more than Accuracy here?**
> A naive model predicting "No Defect" for all samples gets 95% accuracy but misses every defect — which is dangerous in a manufacturing setting!

---

## 📂 Dataset

| File | Dimensions | Description |
|------|-----------|-------------|
| `train.csv` | 1352 × 51 | Training data with labels |
| `test.csv` | 339 × 50 | Test data without labels |
| `solution.ipynb` | — | Complete solution notebook |
| `expected_submission.csv` | 339 × 2 | Final predictions |
| `approach_akanksha.docx` | — | Detailed approach document |

### Variable Description
| Column | Description |
|--------|-------------|
| `CoilID` | Unique identifier for each coil |
| `X1–X49` | Process parameters across multiple stages |
| `Y` | Target: 1 = Alpha Defect, 0 = No Defect |

### Class Distribution
```
No Defect (0): 1286 samples — 95.12%
Defect    (1):   66 samples —  4.88%  ⚠️ Severely imbalanced!
```

---

## 🧠 My Approach

### Step 1 — Data Preprocessing
- Filled missing values using **column median** (robust to outliers)
- 12 columns had missing values (worst: X15 with 160 nulls)
- Applied **RobustScaler** — better than StandardScaler for outlier-heavy industrial data

### Step 2 — Feature Engineering
Added 8 row-level statistical features to capture overall coil behaviour:

```python
df['mean_all']   = row mean of all process parameters
df['std_all']    = row standard deviation
df['max_all']    = row maximum value
df['min_all']    = row minimum value
df['range_all']  = max - min (spread)
df['iqr_all']    = interquartile range (middle 50% spread)
df['n_outliers'] = count of features with z-score > 2
df['cv']         = coefficient of variation (relative variability)
```

> 💡 **Key insight:** A defective coil likely has more extreme or inconsistent readings across its process stages.

### Step 3 — Handling Class Imbalance with SMOTE
```python
SMOTE(random_state=42, sampling_strategy=0.5)
# Creates synthetic defect samples
# 0.5 ratio → defects = 50% of non-defects (not fully balanced)
# Full balancing hurt precision; 0.5 ratio was optimal
```

### Step 4 — Model: XGBoost
```python
XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.01,
    scale_pos_weight=10,   # penalise missing defects 10x more
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=1
)
```

### Step 5 — Threshold Optimisation
```python
# Default threshold 0.5 misses many defects!
# Swept thresholds from 0.01 to 0.99
# Selected highest threshold where Recall = 100%

OPTIMAL_THRESHOLD = 0.870
# Recall = 1.0000 ✅
# Precision = 0.9851 ✅
```

---

## 📊 Results

```
Training Data Evaluation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Recall         : 1.0000  ✅ PASS
  Precision      : 0.9851  ✅ PASS
  False Negatives: 0       ✅ (No defect missed!)
  False Positives: 1       (1 non-defect flagged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test Set Predictions:
  Defects detected: 6 / 339 coils
```

### Confusion Matrix (Training)
```
                 Predicted 0    Predicted 1
Actual 0  (1286)    1285              1
Actual 1  (66  )       0             66
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| Python 3.10 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, metrics |
| imbalanced-learn | SMOTE oversampling |
| XGBoost | Classification model |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AkankshaKesarkar/defect-detection-hot-rolling.git
cd defect-detection-hot-rolling
```

### 2. Install dependencies
```bash
pip install xgboost imbalanced-learn scikit-learn pandas numpy
```

### 3. Run the notebook
```bash
jupyter notebook solution.ipynb
```

### 4. Or run directly in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AkankshaKesarkar/defect-detection-hot-rolling/blob/main/solution.ipynb)

---

## 📁 Project Structure

```
defect-detection-hot-rolling/
│
├── solution.ipynb            # Complete ML solution notebook
├── expected_submission.csv   # Final predictions (339 × 2)
├── approach_akanksha.docx    # Detailed approach document
└── README.md                 # Project documentation
```

---

## 👩‍💻 Author

**Akanksha Ramchandra Kesarkar**
B.E. Computer Science & Engineering | VSM SRKIT, Belagavi | 2024

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/akanksha-kesarkar)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/AkankshaKesarkar)

---

⭐ **If you found this helpful, please star the repo!**
