# Insurance Pricing Forecast using Machine Learning

## Project Overview

This project focuses on predicting medical insurance charges using Machine Learning and advanced regression modeling techniques. The system analyzes customer demographic and health-related attributes such as age, BMI, smoking status, number of children, and region to estimate insurance premium charges.

The project was designed as an end-to-end Machine Learning pipeline including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Log Transformation
- Baseline Modeling
- XGBoost Regression
- Hyperparameter Tuning
- Residual Error Analysis
- Feature Importance Evaluation

The objective was not only to improve prediction accuracy but also to understand the business and statistical factors affecting insurance pricing.

---

# Dataset

Dataset used:
- Medical Cost Personal Dataset

Features:
- Age
- Sex
- BMI
- Number of Children
- Smoking Status
- Region

Target Variable:
- Insurance Charges

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Google Colab

---

# Machine Learning Workflow

## 1. Data Preprocessing

- Loaded and cleaned dataset
- Checked for missing values
- Performed exploratory data analysis
- Identified skewed target distribution

---

## 2. Feature Engineering

Created additional engineered features:

### BMI-Age Interaction
Used to capture nonlinear health-risk relationships.

### Obesity Indicator
Binary feature identifying high BMI customers.

### High Risk Indicator
Combined smoking status and obesity conditions to identify extremely high-risk individuals.

---

## 3. Target Transformation

Insurance charges showed strong right-skewness.

To stabilize variance and improve regression performance, log transformation was applied:

y_log = log(1 + y)

Predictions were inverse transformed during evaluation.

---

## 4. Model Development

### Baseline Model
- Linear Regression

### Advanced Model
- XGBoost Regressor

Hyperparameter optimization was performed using RandomizedSearchCV.

---

# Evaluation Metrics

The model was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Key Insights

## Smoking Status Dominates Insurance Pricing

Exploratory analysis revealed that smoking status had the strongest statistical relationship with insurance charges.

Smokers showed:
- significantly higher median insurance costs
- larger cost variability
- increased risk distribution

This behavior was also reflected in the feature importance analysis where smoking-related features became dominant predictors.

---

## Nonlinear Medical Risk Patterns

The relationship between BMI, age, and charges was nonlinear. Feature engineering improved the model’s ability to capture these interactions.

---

## Residual Analysis

Residual plots indicated:

- strong prediction performance for low and medium insurance charges
- moderate variance for extremely high-cost cases

This is expected behavior in real-world insurance pricing systems due to extreme medical claim variability.

---

# Visualizations

The project includes:
- Insurance charge distribution analysis
- Smoker vs insurance charges boxplot
- Correlation heatmap
- Residual analysis
- Actual vs predicted comparison
- Feature importance visualization

---

# Project Structure

```bash
insurance-pricing-forecast/
│
├── insurance_forecast.ipynb
├── insurance_pricing_model.pkl
├── insurance.csv
├── requirements.txt
├── README.md
│
└── images/
    ├── smoker_vs_charges.png
    ├── residual_analysis.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
