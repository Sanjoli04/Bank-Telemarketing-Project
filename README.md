# Bank Term Deposit Prediction

Machine learning project developed as part of the MLP Project T32024 at IIT Madras.

The objective of this project was to predict whether a customer would subscribe to a bank term deposit based on information collected during direct marketing campaigns.

## Problem Statement

The dataset contains information about customers contacted during bank marketing campaigns. The target variable indicates whether the customer subscribed to a term deposit (`yes` / `no`).

The competition evaluated predictions using the Macro F1 Score.

## Dataset

The dataset contains customer and campaign-related features such as:

- Age
- Job
- Marital status
- Education
- Account balance
- Housing loan
- Personal loan
- Contact type
- Contact duration
- Campaign contacts
- Previous campaign information
- Previous campaign outcome

Target:

- `yes` – customer subscribed to a term deposit
- `no` – customer did not subscribe

## Approach

### 1. Exploratory Data Analysis

Performed exploratory analysis to understand:

- Numerical and categorical variables
- Missing values
- Class distribution
- Feature relationships
- Customer and campaign characteristics

Visualizations were created using Matplotlib and Seaborn.

### 2. Data Preprocessing

The preprocessing pipeline included:

- Handling missing categorical values using `SimpleImputer`
- Encoding categorical variables using `OrdinalEncoder`
- Converting binary variables into numerical representations
- Handling inconsistent values
- Scaling numerical features using `MinMaxScaler`

### 3. Feature Engineering

Created additional features including:

- Days since last contact
- Balance groups
- Housing and personal loan combinations
- Credit/housing/personal loan combinations
- Adult indicator
- Loan indicator
- Previous contact indicator
- Contact frequency
- Contact recency

### 4. Feature Selection

Used `GenericUnivariateSelect` with mutual information to select the most informative features for model training.

### 5. Model Development

Multiple machine learning approaches were explored during the project.

The final model selected for the competition was:

**XGBoost Classifier (`XGBClassifier`)**

The model was evaluated using:

- Macro F1 Score
- Classification Report
- 5-fold Cross Validation

### 6. Hyperparameter Tuning

Used `RandomizedSearchCV` with 5-fold cross-validation to tune the XGBoost model.

Parameters explored included:

- Number of estimators
- Maximum tree depth
- Learning rate
- Subsampling
- Column sampling
- Gamma
- Minimum child weight
- L1 regularization
- L2 regularization

### 7. Prediction

The final model was used to predict the target variable for the test dataset.

Predictions were converted into:

- `yes`
- `no`

and stored along with the corresponding `id` in `submission.csv`.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Optuna
- Jupyter Notebook

## Repository Contents

```text
Bank-Telemarketing-Project/
│
├── README.md
├── 23f1002254-notebook-t32024.ipynb
├── requirements.txt
├── .gitignore
└── submission/
    └── submission.csv
