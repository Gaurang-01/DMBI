# PhishGuard AI

College mini-project for DMBI CA2: machine learning-based phishing website detection.

## Student details

- Name: Gaurang Khanvilkar
- Roll number: 49
- Division: D15C
- Subject: Data Mining and Business Intelligence

## Setup and launch

```powershell
cd C:\Users\Gaurang\Desktop\DMBI
python -m pip install -r requirements.txt
streamlit run app.py
```

If the default Python package server times out while downloading packages, install through this mirror first, then repeat the command above:

```powershell
python -m pip install --no-cache-dir --timeout 120 --retries 2 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

Open the local address Streamlit prints (normally `http://localhost:8501`).

## Project layout

```text
DMBI/
├── app.py                     # Streamlit dashboard
├── config.py                  # Project paths and constants
├── data/                      # Project copy of supplied CSV (unaltered)
├── models/                    # Saved joblib models after dashboard action
├── outputs/                   # Actual saved evaluation table
├── src/                       # Reusable ML, evaluation, prediction, plotting modules
├── notebooks/                 # Analysis notebook
├── requirements.txt
└── README.md
```

## Workflow and CA2 coverage

| CA2 stage | Implementation |
|---|---|
| Real-world problem | Early phishing-risk identification for website security workflows |
| Dataset | Original 10,000-row phishing dataset; `CLASS_LABEL` target |
| Preprocessing | Missing/duplicate checks, ID removal, constant-feature removal, stratified split |
| Data analysis | Class distribution, correlation heatmap, distributions and importance plots |
| Technique and algorithms | Supervised binary classification using Logistic Regression, Decision Tree and Random Forest |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, matrices, reports and ROC curves |
| Insights and conclusion | Feature importance and cybersecurity explanation of false positives/negatives |

The application never changes the original source dataset, does not train on test data, and does not claim to scan live URLs. The dashboard calculates results from current model predictions rather than hardcoding them.
