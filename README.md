# 🤖 AI-Based Customer Support Ticket Auto-Triage System

An AI-powered customer support ticket classification system that automatically analyzes incoming customer messages and predicts the **ticket category, priority, and appropriate support department**.

The system helps support teams reduce manual ticket sorting, improve response time, and route customer issues to the right department efficiently.

## 🚀 Features

* 🤖 Automatic customer ticket classification
* 🏷️ Predicts ticket category
* ⚡ Predicts ticket priority
* 🏢 Recommends support department
* 📝 Natural Language Processing (NLP)
* 📊 Machine Learning-based prediction
* 🔍 Real-time ticket analysis
* 📋 Sample customer ticket examples
* 🎯 Automated ticket routing
* 📈 Prediction results dashboard
* 🌐 Web-based user interface

## 🎯 Problem Statement

Customer support teams receive a large number of tickets every day.

Manually reading and categorizing every ticket can:

* Take significant time
* Delay customer responses
* Cause incorrect routing
* Increase workload for support agents

This project uses **Machine Learning and NLP** to automatically analyze customer messages and assist in routing tickets.

## 💡 Solution

The system takes a customer's message as input and uses trained machine learning models to predict:

```text
Customer Message
       ↓
Text Preprocessing
       ↓
Feature Extraction
       ↓
Machine Learning Model
       ↓
Category Prediction
       ↓
Priority Prediction
       ↓
Department Recommendation
       ↓
Ticket Routing
```

## 🧠 AI/ML Models

The project uses machine learning algorithms for different prediction tasks.

### Category Classification

**Logistic Regression**

Used to classify tickets into categories such as:

* Account
* Payment
* Technical
* Delivery
* Billing
* Product

### Priority Classification

**Multinomial Naive Bayes**

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

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* Multinomial Naive Bayes
* TF-IDF Vectorizer

### Backend

* Flask

### Frontend

* HTML5
* CSS3
* JavaScript

### Database

* PostgreSQL / SQLite

### Development Tools

* VS Code
* Git
* GitHub

## 📂 Project Structure

```text
AI-Customer-Support-Triage/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── customer_support_tickets.csv
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── notebooks/
    └── model_training.ipynb
```

> Update the structure according to your actual project files.

## 📊 Example

### Input

```text
Subject:
Payment Issue

Customer Message:
I made a payment but the amount was deducted from my account and
the order is still showing as unpaid.
```

### AI Result

```text
Category: Payment
Priority: High
Department: Payment Support
```

Another example:

```text
Customer Message:
I cannot log into my account. I have tried resetting my password
multiple times but I still cannot access my account.
```

Result:

```text
Category: Account
Priority: High
Department: Account Support
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kiran-babu-h/AI-Customer-Support-Triage.git
```

### 2. Navigate to the project

```bash
cd AI-Customer-Support-Triage
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000/
```

## 📈 Model Workflow

The training process follows these steps:

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Trained Model
   ↓
Flask Application
   ↓
Real-Time Prediction
```

## 📋 Supported Ticket Categories

The system can be trained to identify categories such as:

| Category  | Example                   |
| --------- | ------------------------- |
| Account   | Login/password problems   |
| Payment   | Payment failed or missing |
| Technical | Application errors        |
| Delivery  | Delayed/missing delivery  |
| Billing   | Invoice or billing issues |
| Product   | Product-related questions |

## ⚡ Priority Levels

| Priority | Description                               |
| -------- | ----------------------------------------- |
| Low      | General questions or minor issues         |
| Medium   | Issue requiring normal support            |
| High     | Important issue requiring quick attention |
| Critical | Urgent issue requiring immediate action   |

## 🏢 Department Routing

Based on the predicted category, the system recommends a suitable department.

```text
Account → Account Support
Payment → Payment Support
Technical → Technical Support
Delivery → Logistics Support
Billing → Billing Support
Product → Product Support
```

## 🎯 Benefits

* Reduces manual ticket classification
* Improves ticket routing
* Saves customer support time
* Helps prioritize urgent issues
* Improves response efficiency
* Provides consistent ticket classification
* Can handle large numbers of incoming tickets

## 🔮 Future Enhancements

* Transformer-based NLP models
* BERT-based ticket classification
* Sentiment analysis
* Automatic response generation
* Email ticket integration
* CRM integration
* Real-time analytics dashboard
* SLA breach prediction
* Multilingual ticket classification
* Human-in-the-loop feedback system

## 👨‍💻 Developer

**Kiran Babu H**

MCA Graduate | Python Developer | AI/ML Enthusiast

### Technical Skills

```text
Python
Machine Learning
NLP
Scikit-learn
Flask
Django
FastAPI
PostgreSQL
SQL
HTML
CSS
JavaScript
Git & GitHub
REST API
```

## 📄 License

This project is developed for **educational, learning, and portfolio purposes**.
