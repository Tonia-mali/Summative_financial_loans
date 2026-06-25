# FinTech Innovations - Machine Learning Loan Approval Model

A binary classification pipeline built on 20,000 historical loan applications, following the CRISP-DM framework.

---

## Files

| File | Description |
|---|---|
| `financial_loan_risk_complete.ipynb` | Main analysis notebook - all code, narrative, and visualizations |
| `Loan.csv` | Dataset (20,000 loan applications, 36 columns) |

---

## Setup

**Requirements:** Python 3.8+

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

Place `Loan.csv` in the same directory as the notebook, then run:
```bash
jupyter notebook financial_loan_risk_complete.ipynb
```

Run all cells top to bottom (Kernel → Restart & Run All).

---

## Project Structure (CRISP-DM)

| Phase | Key Outputs |
|---|---|
| **Business Understanding** | Asymmetric cost analysis ($8k FN / $50k FP), custom Business Cost Score metric, classification approach justification |
| **Data Understanding** | EDA with 8+ visualizations, data quality audit (3 introduced issues identified), feature categorization |
| **Data Preparation** | ColumnTransformer pipeline - separate flows for numerical, ordinal, and nominal features; messy data fixes |
| **Modeling** | 5-algorithm comparison, Logistic Regression selected, GridSearchCV + RandomizedSearchCV tuning |
| **Evaluation** | Test ROC-AUC 0.9943, $6.73M business cost vs $145.7M approve-all baseline, feature importance, segmented fairness analysis |

---

## Results Summary

| Metric | Value |
|---|---|
| ROC-AUC | 0.9943 |
| Precision (approved class) | 0.877 |
| Recall (approved class) | 0.970 |
| F1-Score | 0.921 |
| Business Cost Score (test set) | −$6,732,000 |
| Savings vs. Approve-All baseline | $138,968,000 |

**Top predictive features:** `TotalDebtToIncomeRatio`, `InterestRate`, `EmploymentStatus_Unemployed`, `NetWorth`, `LoanAmount`
