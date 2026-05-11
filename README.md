# 🎓 Student Grade Prediction & Model Interpretability with LIME

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![LIME](https://img.shields.io/badge/LIME-Explainability-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> Predicting whether a student scores above 90 using Logistic Regression and Neural Networks — with local model explanations powered by LIME.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Results & Key Findings](#-results--key-findings)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [References](#-references)

---

## 🔍 Overview

This project combines **predictive modeling** and **model interpretability** on a student performance dataset of ~10,000 records. It answers two questions:

1. Can we accurately predict whether a student scores above 90?
2. *Why* does the model make the predictions it does?

Models trained and compared:
- ✅ Logistic Regression (baseline)
- ✅ Multi-Layer Perceptron (MLP) with hyperparameter tuning
- ✅ LIME explanations for both models

---

## 📂 Dataset

**File:** `grades_dataset.csv` — 10,000 rows, 29 features, 1 binary target

| Category | Features |
|---|---|
| 📖 Academic | `ReadingScore`, `WritingScore` |
| 👤 Demographic | `Gender_female`, `EthnicGroup_group A–E` |
| 👨‍👩‍👧 Family | `NrSiblings`, `ParentEduc_*`, `ParentMaritalStatus_*`, `IsFirstChild_no` |
| 🏃 Lifestyle | `PracticeSport_*`, `WklyStudyHours_*`, `TransportMeans_private` |
| 🏫 School | `LunchType_free/reduced`, `TestPrep_completed` |
| 🎯 **Target** | `Grade>90` — `1` = above 90, `0` = at or below 90 |

> All categorical features are one-hot encoded.

---

## 🔄 Project Workflow

```
Data Loading → Preprocessing → Logistic Regression → Neural Network → LIME Explanations
```

### 1️⃣ Data Loading & Preprocessing
- 60% train / 20% validation / 20% test split via `train_test_split`
- Features standardized with `StandardScaler` (fit on train set only)

### 2️⃣ Logistic Regression
- Trained with scikit-learn defaults
- Evaluated using **Accuracy** and **AUC**
- Model weights visualized as a bar chart for global feature importance

### 3️⃣ Neural Network (MLP Classifier)
- Initial model: `MLPClassifier(max_iter=20000)`
- **Hyperparameter search** over hidden layer sizes:

| Config | Hidden Layers |
|---|---|
| A | `(10,)` |
| B | `(20,)` |
| C | `(10, 20)` |
| D | `(10, 20, 20)` |

- **Extended search** with activation functions: `relu`, `tanh`, `logistic`
- Best model selected by highest **validation AUC**

### 4️⃣ LIME Explanations
- `LimeTabularExplainer` applied to both models
- 10 randomly selected test instances explained
- Settings: `num_features=5`, `num_samples=50,000`
- Linear vs. non-linear explanations compared side by side

---

## 📊 Results & Key Findings

- **LIME validity** — Logistic Regression LIME explanations closely mirror the global weight plot, confirming LIME is working correctly
- **Model complexity** — Neural Network explanations vary more across instances, reflecting captured non-linear relationships
- **Top predictors** — `ReadingScore`, `WritingScore`, `TestPrep_completed`, and `WklyStudyHours` were the most influential features
- **Hyperparameter tuning** — Extended search with varied activations outperformed the default single-layer baseline

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn lime matplotlib
```

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `scikit-learn` | Model training, evaluation, preprocessing |
| `lime` | Local model explanations |
| `matplotlib` | Weight visualization |

---

## 🚀 Usage

1. Clone the repo and ensure `grades_dataset.csv` is in the same directory as the notebook
2. Install dependencies
3. Launch the notebook

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Assignment_1_LIME.ipynb
```

---

## 📚 References

- [LIME Documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_tabular)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016.

---

<p align="center">Made with 🧠 and Python</p>
