# AI-Based Customer Support Ticket Auto-Triage

This is a complete VS Code-ready project using the supplied customer support dataset.

## What the project does

It reads a customer's ticket subject + description and predicts the **Issue Category** automatically.

Categories in this dataset:
- Account
- Billing
- Fraud
- General Inquiry
- Technical

It also maps the predicted category to a support department.

## Dataset

The supplied archive contained:
- `customer_support_tickets.csv` — 20,000 rows
- `enhanced_customer_support_data.csv` — 20,000 rows

The project uses `customer_support_tickets.csv` as the main training dataset. It trains the same 3 models for both **Issue_Category** and **Priority_Level**.

## Three NLP models compared

1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

All three use the same **TF-IDF** text representation so the comparison is fair.

## Current benchmark

The included `outputs/model_comparison.csv` contains the metrics generated from:
- 80% training data
- 20% test data
- stratified split
- random_state=42
- weighted F1 used to select the best model

Best model from this run for category: **Logistic Regression**. See `outputs/priority_model_comparison.csv` for priority results.

## Folder structure

```
customer_support_auto_triage/
├── app.py
├── train_models.py
├── requirements.txt
├── run_windows.bat
├── run_linux_mac.sh
├── data/
│   ├── customer_support_tickets.csv
│   └── enhanced_customer_support_data.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   └── multinomial_naive_bayes.joblib
├── outputs/
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── classification_reports.txt
│   ├── best_model.json
│   ├── predictions/
│   └── confusion_matrices/
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run in VS Code on Windows

Open the project folder in VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000

## Retrain the models

If you change the dataset:

```powershell
python train_models.py
```

This retrains all 3 models and updates:
- model comparison
- classification reports
- model files

## Test example

Subject:
`Payment problem`

Description:
`My payment was deducted but my order was not created.`

The model will predict an issue category based on the learned dataset patterns.

## Important project note

The provided dataset does **not** contain a sentiment column or department column. Therefore, this implementation trains the ML comparison on `Issue_Category` and uses a transparent category-to-department mapping.

Do not claim that sentiment prediction is trained from this dataset unless you add a labeled sentiment dataset or a separate sentiment model.

## Recommended next enhancement

Add separate labeled models for:
- Priority prediction using `Priority_Level`
- Sentiment analysis using a sentiment dataset/model
- SLA breach prediction using `Resolution_Time_Hours`

This will turn the project into a fuller auto-triage system.

## Important result

All three models reached 1.0000 weighted F1 on the supplied dataset in this run. This is unusually high for real customer-support text and likely reflects the synthetic/structured nature of the Kaggle data. Treat this as a dataset benchmark, not as proof that the model will achieve 100% accuracy on real-world tickets.
