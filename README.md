# Finance Course

A repository of exercises and code from the finance course, organized by lecture.

## Structure

```
finance-course/
├── lecture_01/
│   └── uniform_sum_distributions.py   # Sum of Uniform RVs, Monte Carlo vs analytical
├── lecture_02/
│   ├── svm_minimal.py                 # Minimal SVM example via sklearn (every slide concept)
│   ├── svm_from_scratch.py            # Solve the SVM Lagrangian dual explicitly with cvxpy
│   ├── svm_visualize.py               # Step-by-step plot for every stage of svm_from_scratch.py
│   ├── svm_loan_default.py            # Full deep-dive SVM script (commented end-to-end)
│   ├── svm_cds_spreads.py             # ε-SVR for illiquid CDS spread prediction (Halperin Part IV)
│   ├── STRUCTURE.md                   # Mermaid flow charts of the code structure
│   └── SVM_from_scratch_theory.pdf    # Line-by-line theory document (37-page PDF)
└── ...
```

## Setup

```bash
pip install numpy matplotlib scipy scikit-learn pandas cvxpy
```

## Lectures

| Lecture | Topic | Files |
|---------|-------|-------|
| 01 | Probability — Sum of Uniform Random Variables (Irwin-Hall) | `uniform_sum_distributions.py` |
| 02 | Support Vector Machines — Loan Default Prediction + CDS Spread Regression | `svm_minimal.py`, `svm_from_scratch.py`, `svm_visualize.py`, `svm_loan_default.py`, `svm_cds_spreads.py`, [`STRUCTURE.md`](lecture_02/STRUCTURE.md), [`SVM_from_scratch_theory.pdf`](lecture_02/SVM_from_scratch_theory.pdf) |
