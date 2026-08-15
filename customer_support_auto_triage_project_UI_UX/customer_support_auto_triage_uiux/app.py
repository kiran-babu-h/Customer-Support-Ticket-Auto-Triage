from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DATA_FILE = BASE / "data" / "customer_support_tickets.csv"

app = Flask(__name__)

with open(BASE / "outputs" / "best_model.json", "r", encoding="utf-8") as f:
    META = json.load(f)

MODEL_FILE = MODEL_DIR / (META["best_model"].lower().replace(" ", "_") + ".joblib")
MODEL = joblib.load(MODEL_FILE)

priority_df = pd.read_csv(BASE / "outputs" / "priority_model_comparison.csv")
PRIORITY_MODEL_NAME = priority_df.iloc[0]["Model"]
PRIORITY_MODEL = joblib.load(
    MODEL_DIR / ("priority_" + PRIORITY_MODEL_NAME.lower().replace(" ", "_") + ".joblib")
)

DATA = pd.read_csv(DATA_FILE)
DEPARTMENT_MAP = {
    "Billing": "Billing & Payments",
    "Fraud": "Fraud & Security",
    "Technical": "Technical Support",
    "Account": "Account Support",
    "General Inquiry": "Customer Service",
}


def counts(column):
    return {str(k): int(v) for k, v in DATA[column].value_counts().to_dict().items()}


@app.route("/")
def home():
    return render_template("index.html", best_model=META["best_model"])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form
    subject = str(data.get("subject", "")).strip()
    description = str(data.get("description", "")).strip()
    text = f"{subject} {description}".strip()

    if not text:
        return jsonify({"error": "Please enter a ticket subject or description."}), 400

    category = str(MODEL.predict([text])[0])
    priority = str(PRIORITY_MODEL.predict([text])[0])
    department = DEPARTMENT_MAP.get(category, "Customer Service")

    return jsonify({
        "category": category,
        "department": department,
        "priority": priority,
        "model": META["best_model"],
        "priority_model": PRIORITY_MODEL_NAME,
        "message": "Ticket analyzed successfully."
    })


@app.get("/api/dashboard")
def dashboard_data():
    avg_resolution = float(DATA["Resolution_Time_Hours"].mean())
    satisfaction = float(DATA["Satisfaction_Score"].mean())
    return jsonify({
        "stats": {
            "total": int(len(DATA)),
            "high_priority": int(DATA["Priority_Level"].isin(["High", "Critical"]).sum()),
            "avg_resolution": round(avg_resolution, 1),
            "satisfaction": round(satisfaction, 2),
        },
        "categories": counts("Issue_Category"),
        "priorities": counts("Priority_Level"),
        "channels": counts("Ticket_Channel"),
        "models": pd.read_csv(BASE / "outputs" / "model_comparison.csv").to_dict(orient="records"),
        "priority_models": priority_df.to_dict(orient="records"),
        "best_model": META["best_model"],
        "priority_best_model": PRIORITY_MODEL_NAME,
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": META["best_model"]})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
