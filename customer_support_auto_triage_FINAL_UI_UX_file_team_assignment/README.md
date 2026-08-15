# Customer Support Ticket Auto-Triage — UI/UX Edition

A Flask + NLP project that automatically classifies customer support tickets by **issue category**, predicts **priority**, and recommends a support department.

## New UI/UX
- Professional SupportIQ dashboard
- Dark sidebar navigation
- KPI cards
- Category and priority visualizations
- AI ticket analyzer with quick examples
- Category, priority and department prediction cards
- Model comparison tables for 3 models
- Responsive desktop/tablet/mobile layout
- Live dashboard data through Flask API

## Models
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

TF-IDF converts ticket text into numerical features before classification.

## Run in VS Code
**Recommended: Python 3.13** for the pinned dependency versions in `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## Important
The dataset contains 20,000 tickets and fields for category, priority, channel, resolution time and satisfaction. It does not contain a Ticket_Status field, so the dashboard does not invent open/resolved statistics.

The category benchmark is 100% on this supplied dataset. This should be described as a dataset benchmark, not as guaranteed real-world accuracy.
