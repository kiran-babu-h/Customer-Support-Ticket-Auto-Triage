# 🤖 AI-Based Customer Support Ticket Auto-Triage System (UI/UX & Multi-Ticket Edition)

An AI-powered customer support ticket classification system that automatically analyzes incoming customer messages/files and predicts the **ticket category, priority, and appropriate support department**, routing it to the recommended team.

The system helps support teams reduce manual ticket sorting, improve response time, and route customer issues to the right department efficiently, featuring a modern dark dashboard UI and batch multi-ticket file parsing logic.

---

## 🚀 Features

* **🤖 Automatic Customer Ticket Classification**: Predicts ticket category, priority level, and support department recommendation using trained ML models.
* **📂 Multi-Ticket File Analysis**: Upload text/PDF files containing multiple formatted tickets (e.g. separated by headers like `Ticket 1`), slice them automatically, and analyze each ticket individually.
* **💻 Premium UI/UX Dashboard**: Polished dashboard with dark sidebar navigation, dynamic statistics cards, performance charts, interactive triage demo, and instant manual team routing overrides.
* **📊 Model Calibration Page**: Real-time accuracy, precision, recall, and F-1 metric comparison between Logistic Regression, Linear SVM, and Multinomial Naive Bayes models.

---

## 🎯 Problem Statement

Customer support teams receive a large number of tickets every day. Manually reading and categorizing every ticket can:
* Take significant time
* Delay customer responses
* Cause incorrect routing
* Increase workload for support agents

This project uses **Machine Learning and NLP** to automatically analyze customer messages and assist in routing tickets.

---

## 💡 Solution

The system takes a customer's message or document upload as input and uses trained machine learning models to predict:

```text
Customer Message / File Upload
              ↓
      Text Preprocessing
              ↓
      Feature Extraction
              ↓
    Machine Learning Model
              ↓
  Category & Priority Prediction
              ↓
   AI Team Recommendation & Routing
```

---

## 🧠 AI/ML Models

The project uses machine learning algorithms for different prediction tasks.

### Category Classification
**Logistic Regression** (Optimized Category Model)
Used to classify tickets into categories such as:
* Account
* Payment
* Technical
* Delivery
* Billing
* Product

### Priority Classification
**Multinomial Naive Bayes** (Optimized Priority Model)
Used to predict ticket urgency such as:
* Low
* Medium
* High
* Critical

### NLP Pipeline
```text
Customer Message
       ↓
 Text Cleaning
       ↓
  Tokenization
       ↓
TF-IDF Vectorization
       ↓
Machine Learning Model
       ↓
   Prediction
```

---

## 🛠️ Technologies Used

### Programming Language
* Python

### Machine Learning
* Scikit-learn (Logistic Regression, Multinomial Naive Bayes, Linear SVM)
* TF-IDF Vectorizer
* Joblib (Model serialization)

### Web Framework & Frontend
* Flask
* HTML5, CSS3 (Vanilla), JavaScript (ES6)
* Chart.js (Data visualizations)

---

## 📂 Project Structure

```text
Customer-Support-Ticket-Auto-Triage/
│
├── app.py                      # Flask Application Server
├── train_models.py             # ML Model Training Pipeline
├── requirements.txt            # Python Dependencies
├── README.md                   # Documentation
│
├── data/                       # Dataset Files
│   ├── customer_support_tickets.csv
│   └── enhanced_customer_support_data.csv
│
├── models/                     # Trained Serialized Joblib Models
│   ├── linear_svm.joblib
│   ├── logistic_regression.joblib
│   └── multinomial_naive_bayes.joblib
│
├── outputs/                    # Training performance reports & metrics
│   ├── best_model.json
│   ├── classification_reports.txt
│   ├── model_comparison.csv
│   └── priority_model_comparison.csv
│
├── templates/                  # Frontend HTML Templates
│   └── index.html
│
└── static/                     # Styling stylesheet and charts assets
    └── style.css
```

---

## ⚙️ Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/kiran-babu-h/Customer-Support-Ticket-Auto-Triage.git
cd Customer-Support-Ticket-Auto-Triage
```

### 2. Create a virtual environment
**Recommended: Python 3.13** or Python 3.10+
```bash
python -m venv .venv
```

### 3. Activate the environment
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 4. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Run the application
```bash
python app.py
```
Open: `http://127.0.0.1:5000/` in your browser.

---

## 📈 Model Workflow

The training process follows these steps:
```text
Dataset → Data Cleaning → Text Preprocessing → Train/Test Split → TF-IDF Vectorization → Model Training → Model Evaluation → Save Joblib Models → Flask Web Service
```

---

## 👨‍💻 Developer

**Kiran Babu H**  
*MCA Graduate | Python Developer | AI/ML Enthusiast*

### Technical Skills
```text
Python | Machine Learning | NLP | Scikit-learn | Flask | Django | FastAPI | PostgreSQL | SQL | HTML | CSS | JavaScript | Git & GitHub | REST API
```

---

## 📄 License

This project is developed for **educational, learning, and portfolio purposes**.
