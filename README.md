# 🏦 LoanAdmin – Loan Approval Prediction System

An end-to-end Machine Learning project that predicts loan approval eligibility using applicant financial information. The project combines a Scikit-learn classification pipeline with a web-based dashboard for loan prediction and application history management.

## 📁 Project Architecture

```text
Loan-Approval-Prediction/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── loan_approval_dataset.csv
│
├── models/
│   └── loan_approval_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── predictor.py
│   └── database.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── dashboard.css
│   └── js/
│       └── dashboard.js
│
├── notebook/
│   ├── train_model.py
│   └── EDA_and_Model_Building.ipynb
│
└── tests/
    └── test_core.py
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Adarsh8158/Loan-Approval-Prediction.git

cd Loan-Approval-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application

```bash
python app.py
```

Open your browser:

http://127.0.0.1:5000

### Alternative: Streamlit Application

```bash
streamlit run streamlit_app.py
```

## 🎯 Key Features

### 1. Machine Learning Prediction

- Predicts loan approval eligibility using a Random Forest Classifier.
- Uses a Scikit-learn Pipeline for preprocessing and classification.
- Handles numerical and categorical applicant information.
- Provides predicted approval status and probability.
- Calculates estimated monthly EMI.

### 2. Applicant Information Processing

The prediction system accepts:

- Number of dependents
- Education
- Self-employment status
- Annual income
- Loan amount
- Loan term
- CIBIL score
- Residential assets value
- Commercial assets value
- Luxury assets value
- Bank asset value

### 3. Web Dashboard

- Dark-themed dashboard interface.
- Applicant prediction form.
- Loan prediction results.
- Applicant prediction history.
- Dashboard charts and visual summaries.

### 4. Database Management

- SQLite database integration.
- Stores prediction history.
- Retrieves recent prediction records.
- Supports clearing prediction history.

### 5. Model Development

- Exploratory Data Analysis (EDA).
- Data preprocessing.
- Missing-value handling.
- Feature encoding and scaling.
- Random Forest model training.
- Model evaluation.
- Model serialization using Joblib.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Web Framework | Flask, Streamlit |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Database | SQLite |
| Model Storage | Joblib |
| Development | Jupyter Notebook, VS Code |

## 📊 Dataset Information

- **Dataset:** Loan Approval Prediction Dataset
- **Total Records:** 4,269
- **Target Variable:** `loan_status`
- **Prediction Classes:** Approved and Rejected

The dataset contains applicant financial information, credit scores, employment details, and asset values.

## 🧠 Machine Learning Workflow

1. Load the loan approval dataset.
2. Perform Exploratory Data Analysis.
3. Clean and preprocess the data.
4. Separate features and target variable.
5. Split data into training and testing sets.
6. Apply numerical scaling and categorical encoding.
7. Train a Random Forest Classifier.
8. Evaluate model performance.
9. Save the trained pipeline.
10. Use the saved model for new predictions.

## 📈 Model Evaluation

The model is evaluated using the following metrics:

- Accuracy Score
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Evaluation metrics should be reported from the latest reproducible model training run.

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes. Its predictions are based on a historical dataset and should not be considered actual financial advice or used as the sole basis for real-world loan approval decisions.

## 👨‍💻 Author

**Adarsh Yadav**

- GitHub: [Adarsh8158](https://github.com/Adarsh8158)
- LinkedIn: [Adarsh Yadav](https://www.linkedin.com/in/adarsh-yadav-49225140b)

