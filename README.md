# Netflix Customer Intelligence, Churn Prediction & AI Analytics Copilot

An end-to-end customer intelligence platform modeled on a Netflix-style subscription business. The project combines data cleaning, PostgreSQL data engineering, SQL analytics, exploratory data analysis, machine learning, and an AI-powered analytics copilot into one complete workflow.

## What's Inside

| Layer | Deliverable | Status |
|---|---|---|
| Data Cleaning | `src/data_processing/clean.py` — deduplication, casing, dates, outlier handling, subscription ID reconciliation | ✅ Complete |
| Database | `database/schema.sql` + `database/load_data.py` — 9 normalized tables with PK/FK constraints and indexes | ✅ Complete |
| SQL Analytics | `database/sql_business_questions.sql` — 37 business questions across A/B/C tiers | ✅ 37/37 tested |
| EDA | `src/eda/eda_plots.py` — 30 visualizations across the PRD themes | ✅ Complete |
| Feature Engineering | `src/features/build_features.py` — customer-level behavioral and subscription features | ✅ Complete |
| Machine Learning | `src/models/train_models.py` — churn classification, customer segmentation, and CLV prediction | ✅ Complete |
| AI Analytics Copilot | `src/llm/copilot.py` — 6 capabilities including text-to-SQL, RAG, explanations, reports, root-cause analysis, and retention recommendations | ✅ Tested |
| Streamlit Application | `app/app.py` — 7 interactive application pages | ✅ Complete |

## Key Results

- **8,000 customers** after data cleaning
- **9 relational PostgreSQL tables**
- **37/37 SQL business questions tested successfully**
- **30 EDA visualizations**
- **30 customer-level features**
- **24.89% dataset churn rate**
- **6 AI Copilot capabilities**
- **7 Streamlit application pages**
- Best churn model: **Logistic Regression — AUC 0.6414**
- Best CLV model: **Random Forest — R² 0.998**
- Customer segmentation: **7 K-Means clusters**

---

# Quick Start — Windows

## 1. Prerequisites

Install the following:

- Python 3.11
- PostgreSQL 17
- VS Code

Open the project folder in VS Code and open a **PowerShell terminal** in the project root.

## 2. Create the Python Virtual Environment

Run:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 4. Configure PostgreSQL

The application expects PostgreSQL with:

| Setting | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `netflix` |
| Username | `postgres` |
| Password | `postgres` |

Make sure the PostgreSQL service is running.

On Windows, you can check it with:

```powershell
Get-Service *postgres*
```

If PostgreSQL 17 is installed as a Windows service, it should show a status of `Running`.

## 5. Create the Database

If the `netflix` database does not already exist, run:

```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -p 5432 -d postgres -c "CREATE DATABASE netflix;"
```

If PostgreSQL asks for the password, enter:

```text
postgres
```

If you receive:

```text
ERROR: database "netflix" already exists
```

that is fine. The database has already been created.

## 6. Clean the Data

Run:

```powershell
python src/data_processing/clean.py
```

This creates the cleaned datasets inside:

```text
data/processed/
```

## 7. Load Data into PostgreSQL

Run:

```powershell
python database/load_data.py
```

The loader creates and populates the 9 database tables from the processed datasets.

## 8. Run the SQL Tests

Run:

```powershell
python tests/test_sql_queries.py
```

Expected result:

```text
PASSED: 37/37
```

## 9. Build Customer Features

Run:

```powershell
python src/features/build_features.py
```

This creates:

```text
data/processed/customer_features.csv
```

## 10. Train the Machine Learning Models

Run:

```powershell
python src/models/train_models.py
```

The trained models are saved inside:

```text
models/
```

The pipeline includes:

- Churn classification
- Customer segmentation using K-Means
- Customer Lifetime Value prediction

## 11. Launch the Streamlit Application

Run:

```powershell
python -m streamlit run app/app.py
```

Streamlit will provide a local address similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

# Application Pages

The Streamlit application contains seven main sections:

### 1. Executive Dashboard

Provides a high-level view of:

- Total customers
- Active customers
- Churn rate
- Monthly recurring revenue
- Customer satisfaction
- Subscription performance

### 2. Customer Analytics

Explores:

- Customer demographics
- Countries
- Subscription plans
- Acquisition channels
- Device usage
- Viewing behavior
- Popular content

### 3. Churn Prediction

Uses the trained machine learning model to estimate customer churn probability.

The page allows customer-level analysis and provides a predicted churn risk.

### 4. Customer Segmentation

Uses K-Means clustering to group customers according to behavioral and subscription characteristics.

The resulting segments include groups such as:

- High-Value Loyal
- At-Risk Low-Engagement
- Moderate Engaged
- Price-Sensitive New Signups

### 5. AI Analytics Copilot

The analytics copilot provides six capabilities:

1. Natural-language to SQL
2. Retrieval-Augmented Generation (RAG)
3. Churn explanations
4. Automated reports
5. Root-cause analysis
6. Retention recommendations

The copilot can operate using the project's deterministic offline engine, so an external API key is not required for the core application.

### 6. Support Intelligence

Uses customer support information to provide insights into:

- Support issues
- Customer satisfaction
- Common problems
- Customer-level support context

### 7. Automated Reports

Generates automated analytical summaries and business reports using the project's analytics and AI components.

---

# Machine Learning

## Churn Classification

Four classification models are evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

The best-performing model in the current dataset is:

**Logistic Regression**

- AUC: **0.6414**
- Precision: **0.337**
- Recall: **0.628**
- F1 Score: **0.439**

## Customer Segmentation

K-Means clustering is used to identify customer groups.

Current configuration:

- Number of clusters: **7**
- Silhouette score: **0.181**

## Customer Lifetime Value

Three regression approaches are evaluated:

- Linear Regression
- Random Forest
- XGBoost

The current best model is:

**Random Forest**

- MAE: **2.29**
- RMSE: **4.44**
- R²: **0.998**

---

# Project Structure

```text
customer-intelligence-platform/
│
├── app/
│   ├── app.py
│   ├── db.py
│   └── pages/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── database/
│   ├── schema.sql
│   ├── load_data.py
│   └── sql_business_questions.sql
│
├── models/
│   ├── churn_model.pkl
│   ├── churn_metrics.json
│   ├── segment_model.pkl
│   ├── segment_labels.json
│   ├── clv_model.pkl
│   └── clv_metrics.json
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data_processing/
│   │   └── clean.py
│   ├── eda/
│   │   └── eda_plots.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train_models.py
│   └── llm/
│       ├── copilot.py
│       └── llm_client.py
│
├── tests/
│   └── test_sql_queries.py
│
├── README.md
└── requirements.txt
```

---

# Data Pipeline

The project follows an end-to-end analytics workflow:

```text
Raw CSV Data
     ↓
Data Cleaning
     ↓
Processed CSV Data
     ↓
PostgreSQL Database
     ↓
SQL Analytics
     ↓
EDA & Visualization
     ↓
Feature Engineering
     ↓
Machine Learning
     ↓
AI Analytics Copilot
     ↓
Streamlit Dashboard
```

---

# Database

The PostgreSQL database contains nine relational tables:

1. `customers`
2. `subscription_plans`
3. `subscriptions`
4. `content`
5. `viewing_activity`
6. `payments`
7. `support_tickets`
8. `customer_feedback`
9. `churn_labels`

The database includes primary keys, foreign keys, and indexes designed to support analytical queries.

---

# AI Analytics Copilot

The AI Copilot is designed to make analytics accessible through natural language.

Examples of questions include:

```text
What is the average customer satisfaction score?
```

```text
Which subscription plan has the highest revenue?
```

```text
What are the main drivers of customer churn?
```

```text
How can we improve retention for high-risk customers?
```

The project includes an offline deterministic engine so the application can be demonstrated without requiring an external LLM API.

An OpenAI-compatible API can optionally be configured using:

```text
OPENAI_API_KEY
```

---

# Testing

The project includes automated validation for the SQL analytics layer.

Run:

```powershell
python tests/test_sql_queries.py
```

Current result:

```text
37/37 business questions passed
```

The database loading pipeline also performs foreign-key orphan checks after loading the data.

---

# Technologies Used

### Programming & Analytics

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

### Database

- PostgreSQL
- SQL
- Psycopg2
- SQLAlchemy

### Visualization

- Matplotlib
- Seaborn
- Streamlit

### Machine Learning

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- K-Means Clustering
- Regression models

### AI

- Natural-language analytics
- Text-to-SQL
- RAG
- LLM integration
- Offline deterministic analytics engine

---

# Notes for Reviewers

This project is designed to demonstrate a complete analytics engineering and machine learning workflow rather than only an isolated model.

The recommended execution order is:

```text
1. Install dependencies
2. Configure PostgreSQL
3. Create the netflix database
4. Run data cleaning
5. Load data into PostgreSQL
6. Run SQL tests
7. Build customer features
8. Train ML models
9. Launch Streamlit
```

The trained model files and processed datasets are included so the application can be reviewed without having to regenerate every artifact first.

---

# Author

**Talha Patel**

Built as an end-to-end customer analytics, machine learning, and AI analytics engineering project.