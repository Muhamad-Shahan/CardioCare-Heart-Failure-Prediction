# 🫀 CardioCare: Heart Failure Risk Prediction AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app/)
[![Medical AI](https://img.shields.io/badge/Domain-Healthcare-red.svg)](https://pubmed.ncbi.nlm.nih.gov/)
[![Model](https://img.shields.io/badge/Model-Support_Vector_Machine-blue.svg)](https://scikit-learn.org/)

## 📋 Abstract
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early stratification of high-risk patients is critical for effective intervention. 

**CardioCare** is a machine learning-based **Clinical Decision Support System (CDSS)** designed to predict mortality events in heart failure patients. This project implements a **Comparative Kernel Analysis** using Support Vector Machines (SVM), evaluating the efficacy of Linear, Polynomial, and Radial Basis Function (RBF) kernels on clinical data.

> **[🔴 Launch Clinical Dashboard](https://your-app-link.streamlit.app/)**

## 🔬 Scientific Methodology
The model was developed using the **UCI Heart Failure Clinical Records** dataset (N=299).

### 1. Preprocessing & Feature Engineering
* **Standardization:** Applied `StandardScaler` to normalize features (e.g., Platelets vs. Serum Creatinine), which is mathematically critical for SVM distance hyperplanes.
* **Leakage Prevention:** The `time` variable (follow-up period) was rigorously excluded during training, as it constitutes target leakage in a predictive setting.

### 2. Algorithmic Approach
* **Algorithm:** Support Vector Classifier (SVC).
* **Class Balancing:** Implemented `class_weight='balanced'` to penalize false negatives, addressing the dataset's imbalance (Death events are the minority class).
* **Hyperparameter Tuning:** `GridSearchCV` was used to optimize $C$ (Regularization) and $\gamma$ (Kernel Coefficient).

### 3. Kernel Comparison Results
| Kernel | Accuracy | Precision | Recall (Death) |
|:------:|:--------:|:---------:|:--------------:|
| **RBF** | **~77%** | **High** | **Balanced** |
| Linear | ~74% | Moderate | Low |
| Poly | ~72% | Low | Low |

*Conclusion:* The **RBF Kernel** demonstrated superior capacity to model the non-linear decision boundaries of physiological data.

## 📊 Key Predictive Features
The model identified the following biomarkers as most significant for mortality risk:
1.  **Serum Creatinine:** Indicator of renal function.
2.  **Ejection Fraction:** Percentage of blood leaving the heart.
3.  **Age:** Advanced age correlates with higher risk.

## 💻 Installation & Usage

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repository
git clone [https://github.com/Muhammad-Shahan/CardioCare-Heart-Failure-Prediction.git](https://github.com/Muhammad-Shahan/CardioCare-Heart-Failure-Prediction.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Application
streamlit run app.py
