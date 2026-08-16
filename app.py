from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import io, re, csv
from pathlib import Path
import json
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DATA_FILE = BASE / "data" / "customer_support_tickets.csv"

app = Flask(__name__)

UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "csv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_uploaded_text(file):
    ext = file.filename.rsplit(".", 1)[1].lower()
    raw = file.read()
    if ext == "txt":
        return raw.decode("utf-8", errors="ignore")
    if ext == "csv":
        rows = csv.reader(io.StringIO(raw.decode("utf-8", errors="ignore")))
        return "\n".join(" | ".join(row) for row in rows)
    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError("PDF reading failed. Install pypdf.")
    if ext == "docx":
        try:
            from docx import Document
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            raise ValueError("DOCX reading failed. Install python-docx.")
    return ""


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

ASSIGNMENT_TEAMS = [
    {"name": "Billing Team", "description": "Payments, refunds and billing issues"},
    {"name": "Technical Team", "description": "Website, app and technical issues"},
    {"name": "Delivery Team", "description": "Delivery, shipment and order issues"},
    {"name": "Self Review / AI Team", "description": "Normal review and AI-assisted tickets"},
]

def recommend_team(category, text):
    value=(str(category)+" "+str(text)).lower()
    if any(w in value for w in ["payment","billing","refund","charged","transaction"]): return "Billing Team"
    if any(w in value for w in ["delivery","shipment","arrived","late","courier","tracking"]): return "Delivery Team"
    if any(w in value for w in ["error","upload","login","password","website","app","technical","bug"]): return "Technical Team"
    return "Self Review / AI Team"

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

    low = text.lower()
    signals = []
    if any(w in low for w in ["refund", "payment", "charged", "transaction", "billing"]):
        signals.append("payment/billing language")
    if any(w in low for w in ["urgent", "immediately", "asap", "critical", "blocked"]):
        signals.append("urgency language")
    if any(w in low for w in ["error", "failed", "cannot", "can't", "unable", "not working"]):
        signals.append("failure/error language")
    if any(w in low for w in ["late", "delay", "not arrived", "missing"]):
        signals.append("delivery/delay language")
    
    if "payment failed" in low and ("money was not available" in low or "account" in low or "insufficient" in low):
        reason = "Payment failure caused by insufficient account balance."
    elif "payment" in low or "billing" in low or "refund" in low:
        reason = "Payment-related issue detected."
    else:
        reason = ", ".join(signals[:3]) or "Ticket content and learned text patterns"

    return jsonify({
        "category": category,
        "department": department,
        "priority": priority,
        "reason": reason,
        "recommended_team": recommend_team(category, text),
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



def split_tickets(text):
    # Detect Ticket 1, Ticket #1, Ticket 1 -, Ticket #1:
    pattern = r'(?=Ticket\s*(?:#\s*)?\d+\s*[-–—:])'
    sections = re.split(pattern, text, flags=re.IGNORECASE)
    tickets = []
    for section in sections:
        section = section.strip()
        if re.match(r'^Ticket\s*(?:#\s*)?\d+\s*[-–—:]', section, re.IGNORECASE):
            tickets.append(section)
    return tickets


def get_field(text, field, next_fields):
    stop = "|".join(next_fields)
    pattern = rf"{field}\s*:\s*(.*?)(?=\n(?:{stop})\s*:|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


@app.post("/analyze-file")
def analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "Please select a file."}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Supported files: PDF, DOCX, TXT and CSV."}), 400
    try:
        raw_text = extract_uploaded_text(file)
        if not raw_text or not raw_text.strip():
            return jsonify({"error": "No readable text was found in the file."}), 400
        
        # Check if the file contains multiple tickets
        tickets_text = split_tickets(raw_text)
        
        if len(tickets_text) > 0:
            # Multi-ticket mode
            results = []
            for i, ticket in enumerate(tickets_text, 1):
                subject = get_field(ticket, "Subject", ["Description", "Ticket ID", "Customer"])
                description = get_field(ticket, "Description", ["Ticket ID", "Customer", "Subject"])
                
                combined = f"{subject} {description}".strip()
                if not combined:
                    combined = ticket.strip()
                
                # Predict category and priority using ML models
                category = str(MODEL.predict([combined])[0])
                priority = str(PRIORITY_MODEL.predict([combined])[0])
                department = DEPARTMENT_MAP.get(category, "Customer Service")
                
                low = combined.lower()
                signals = []
                if any(w in low for w in ["refund", "payment", "charged", "transaction", "billing"]):
                    signals.append("payment/billing language")
                if any(w in low for w in ["urgent", "immediately", "asap", "critical", "blocked"]):
                    signals.append("urgency language")
                if any(w in low for w in ["error", "failed", "cannot", "can't", "unable", "not working"]):
                    signals.append("failure/error language")
                if any(w in low for w in ["late", "delay", "not arrived", "missing"]):
                    signals.append("delivery/delay language")
                
                if "payment failed" in low and ("money was not available" in low or "account" in low or "insufficient" in low):
                    reason = "Payment failure caused by insufficient account balance."
                elif "payment" in low or "billing" in low or "refund" in low:
                    reason = "Payment-related issue detected."
                else:
                    reason = ", ".join(signals[:3]) or "Ticket content and learned text patterns"
                
                results.append({
                    "ticket_number": i,
                    "subject": subject or f"Ticket #{i}",
                    "description": description or ticket[:200].strip() + ("..." if len(ticket) > 200 else ""),
                    "category": category,
                    "priority": priority,
                    "department": department,
                    "reason": reason,
                    "recommended_team": recommend_team(category, combined),
                    "model": META["best_model"],
                    "priority_model": PRIORITY_MODEL_NAME
                })
            
            return jsonify({
                "success": True,
                "is_multi": True,
                "ticket_count": len(results),
                "results": results
            })
            
        else:
            # Single-ticket mode
            text = re.sub(r"\s+", " ", raw_text).strip()[:20000]
            category = str(MODEL.predict([text])[0])
            priority = str(PRIORITY_MODEL.predict([text])[0])
            department = DEPARTMENT_MAP.get(category, "Customer Service")
            
            low = text.lower()
            signals = []
            if any(w in low for w in ["refund", "payment", "charged", "transaction", "billing"]):
                signals.append("payment/billing language")
            if any(w in low for w in ["urgent", "immediately", "asap", "critical", "blocked"]):
                signals.append("urgency language")
            if any(w in low for w in ["error", "failed", "cannot", "can't", "unable", "not working"]):
                signals.append("failure/error language")
            if any(w in low for w in ["late", "delay", "not arrived", "missing"]):
                signals.append("delivery/delay language")
            
            if "payment failed" in low and ("money was not available" in low or "account" in low or "insufficient" in low):
                reason = "Payment failure caused by insufficient account balance."
            elif "payment" in low or "billing" in low or "refund" in low:
                reason = "Payment-related issue detected."
            else:
                reason = ", ".join(signals[:3]) or "Ticket content and learned text patterns"
                
            result = {
                "ticket_number": 1,
                "subject": "Uploaded File Content",
                "description": text[:500] + ("..." if len(text) > 500 else ""),
                "category": category,
                "priority": priority,
                "department": department,
                "reason": reason,
                "recommended_team": recommend_team(category, text),
                "model": META["best_model"],
                "priority_model": PRIORITY_MODEL_NAME
            }
            
            return jsonify({
                "success": True,
                "is_multi": False,
                "ticket_count": 1,
                "results": [result],
                "filename": secure_filename(file.filename),
                "characters_analyzed": len(text),
                "category": category,
                "priority": priority,
                "department": department,
                "recommended_team": recommend_team(category, text),
                "model": META["best_model"],
                "priority_model": PRIORITY_MODEL_NAME,
                "reason": reason
            })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Could not analyze file: {e}"}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": META["best_model"]})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
