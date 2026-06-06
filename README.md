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
│   ├── svm_loan_default.py            # Full deep-dive SVM script (commented end-to-end)
│   └── STRUCTURE.md                   # Mermaid flow charts of the code structure
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
| 02 | Support Vector Machines — Loan Default Prediction (SVC + SVR, kernel trick, KKT) | `svm_minimal.py`, `svm_from_scratch.py`, `svm_loan_default.py`, [`STRUCTURE.md`](lecture_02/STRUCTURE.md) |
