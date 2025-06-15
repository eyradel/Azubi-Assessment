# Term Deposit Subscription Predictor

This repository contains a complete data analytics pipeline and a interactive web app to predict whether a client will subscribe to a bank term deposit. It includes data exploration, model training, artifact serialization, and a Streamlit-based user interface.

---

##  Project Structure

```
├── data/
│   ├── bank-additional-full.csv   # Full dataset (41,188 rows, 20 features)
│   ├── bank-additional.csv        # 10% sample of full dataset (4,119 rows)
│   └── README.md                  # (This file)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: distributions, correlations, visualizations
│   ├── 02_feature_engineering.ipynb # Feature creation, encoding, scaling
│   └── 03_model_training.ipynb    # Model training, evaluation, class-imbalance handling
│
├── artifacts/
│   ├── term_deposit_model.pkl     # Serialized trained model
│   ├── scaler.pkl                 # Serialized StandardScaler object
│   └── columns.pkl                # List of feature columns used for encoding
│
├── app/
│   ├── app.py                     # Streamlit application for predictions
│   └── requirements.txt           # Python packages needed for the app
│
└── README.md                      # Project overview and instructions
```

---

##  Data Description

The banking marketing dataset (`bank-additional-full.csv`) includes client and campaign attributes:

* **Numeric features**: `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
* **Categorical features**: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`
* **Target**: `y` (`yes` / `no`) indicates subscription to a term deposit.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/term-deposit-predictor.git
   cd term-deposit-predictor
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r app/requirements.txt
   ```

4. **Download data**
   Place `bank-additional-full.csv` and `bank-additional.csv` under the `data/` folder.

---


##  Running the Streamlit App

1. **Ensure artifacts are in place** under the `artifacts/` directory:

   * `term_deposit_model.pkl`
   * `scaler.pkl`
   * `columns.pkl`

2. **Copy or link these files** into the `app/` directory.

3. **Launch the app**

   ```bash
   cd app
   streamlit run app.py
   ```

4. **Interact**

   * Upload a CSV (semicolon- or comma-separated) of client features.
   * View predictions and probabilities.
   * Download the results as a new CSV.

---

##  Customization

* Adjust model parameters or swap in more advanced algorithms in `03_model_training.ipynb`.
* Modify UI styling or layout in `app/app.py` (e.g., navbar, theme).
* Extend feature engineering (e.g., interaction terms, target encoding).

---
