"""
Retrain and compare 3 NLP models for Issue_Category and Priority_Level:
1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes

Run:
    python train_models.py
"""

from pathlib import Path
import json, warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "customer_support_tickets.csv"
MODEL_DIR = BASE / "models"
OUT = BASE / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
(OUT / "confusion_matrices").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["text"] = (
    df["Ticket_Subject"].fillna("").astype(str) + " " +
    df["Ticket_Description"].fillna("").astype(str)
).str.replace(r"\s+", " ", regex=True).str.strip()

def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, sublinear_tf=True, max_features=60000)),
            ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, sublinear_tf=True, max_features=60000)),
            ("clf", LinearSVC(C=1.5, class_weight="balanced"))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=.98, sublinear_tf=True, max_features=60000)),
            ("clf", MultinomialNB(alpha=.15))
        ])
    }

# 1. Category Models
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["Issue_Category"],
    test_size=0.20, random_state=42, stratify=df["Issue_Category"]
)

models = get_models()
rows = []
report_text = ""
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    rows.append([name, accuracy_score(y_test, pred), p, r, f1])
    report_text += f"\n{'='*70}\n{name}\n" + classification_report(y_test, pred, digits=4, zero_division=0)
    joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

with open(OUT / "classification_reports.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

comparison = pd.DataFrame(rows, columns=["Model","Accuracy","Precision_Weighted","Recall_Weighted","F1_Weighted"])
comparison = comparison.sort_values("F1_Weighted", ascending=False)
comparison.to_csv(OUT / "model_comparison.csv", index=False)
print("\nCATEGORY MODEL COMPARISON")
print(comparison.to_string(index=False))

best_category_model = comparison.iloc[0]["Model"]
print("\nBest Category Model:", best_category_model)

best_model_info = {
    "best_model": best_category_model,
    "selection_metric": "weighted F1",
    "dataset_rows": len(df),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "target": "Issue_Category",
    "classes": sorted(df["Issue_Category"].dropna().unique().tolist()),
    "text_fields": ["Ticket_Subject", "Ticket_Description"],
    "random_state": 42
}
with open(OUT / "best_model.json", "w", encoding="utf-8") as f:
    json.dump(best_model_info, f, indent=2)

# 2. Priority Models
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    df["text"], df["Priority_Level"],
    test_size=0.20, random_state=42, stratify=df["Priority_Level"]
)

priority_models = get_models()
p_rows = []
p_report_text = ""
for name, model in priority_models.items():
    model.fit(X_train_p, y_train_p)
    pred = model.predict(X_test_p)
    p, r, f1, _ = precision_recall_fscore_support(y_test_p, pred, average="weighted", zero_division=0)
    p_rows.append([name, accuracy_score(y_test_p, pred), p, r, f1])
    p_report_text += f"\n{'='*70}\n{name}\n" + classification_report(y_test_p, pred, digits=4, zero_division=0)
    joblib.dump(model, MODEL_DIR / f"priority_{name.lower().replace(' ', '_')}.joblib")

with open(OUT / "priority_classification_reports.txt", "w", encoding="utf-8") as f:
    f.write(p_report_text)

p_comparison = pd.DataFrame(p_rows, columns=["Model","Accuracy","Precision_Weighted","Recall_Weighted","F1_Weighted"])
p_comparison = p_comparison.sort_values("F1_Weighted", ascending=False)
p_comparison.to_csv(OUT / "priority_model_comparison.csv", index=False)
print("\nPRIORITY MODEL COMPARISON")
print(p_comparison.to_string(index=False))
print("\nBest Priority Model:", p_comparison.iloc[0]["Model"])
