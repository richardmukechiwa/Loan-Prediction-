# Loan Default Prediction

## Project Overview
This project aims to predict loan default risk based on borrower characteristics and loan information. By analyzing and modeling the dataset, we explore factors that may indicate a high likelihood of default and develop a predictive model for risk assessment.

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Project Workflow](#project-workflow)
  - [Import Libraries](#import-libraries)
  - [Data Loading and Exploration](#data-loading-and-exploration)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Modeling](#modeling)
  - [Model Evaluation](#model-evaluation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Feature Importance](#feature-importance)
  - [Model Saving](#model-saving)
- [Results](#results)
- [Conclusion](#conclusion)
- [Files in the Repository](#files-in-the-repository)
- [Instructions for Running the Notebook](#instructions-for-running-the-notebook)
- [Future Work](#future-work)

---

## Dataset Overview
The dataset used in this project, `credit_risk_dataset.csv`, contains information about various loan applicants and associated characteristics. It includes features like income, loan amount, loan purpose, and previous credit history, which are analyzed to predict the likelihood of loan default.

## Project Workflow

### Import Libraries
The following libraries were used:
- `pandas` for data manipulation
- `matplotlib` and `seaborn` for visualization
- `sklearn` for preprocessing, model building, and evaluation
- `xgboost` for implementing the XGBoost algorithm

### Data Loading and Exploration
The dataset was loaded and inspected to understand its structure, including a review of initial rows, feature names, data types, and shape. Basic data checks were conducted to identify missing values and potential anomalies.

### Data Preprocessing
To prepare the data for analysis:
- **Missing Values**: Handled using `SimpleImputer` to fill missing values in both numerical and categorical columns.
- **Encoding**: Categorical features were encoded using `LabelEncoder` and `OneHotEncoder` where appropriate.
- **Scaling**: Numerical features were standardized to ensure consistency across model training.

### Exploratory Data Analysis (EDA)
Visualizations were generated to better understand data distributions and relationships:
- **Count Plots** for categorical variables such as loan intent.
- **Box Plots and Scatter Plots** to analyze relationships between features, such as income vs. loan amount.
- **Correlation Analysis** to highlight significant relationships.

### Modeling
Three primary models were trained to predict loan defaults:
1. **Logistic Regression**: Provides a baseline prediction model.
2. **XGBoost**: Gradient boosting model chosen for its performance and ability to handle imbalanced data.
3. **Support Vector Machine (SVM)**: Explored for its classification capabilities.

### Model Evaluation
Each model was evaluated using various metrics:
- **Accuracy Score**: To assess the overall prediction rate.
- **Classification Report**: To examine precision, recall, and F1-score.
- **Confusion Matrix**: To visualize true vs. false predictions.
- **ROC-AUC Score**: To measure the model's discriminative ability.

### Hyperparameter Tuning
`GridSearchCV` was used to optimize model parameters, identifying the best configurations to improve performance.

### Feature Importance
Feature importance was derived using XGBoost, which highlighted key predictors in loan default, allowing for an interpretive view of model decisions.

### Model Saving
The final model was saved using the `pickle` library to create a serialized file (`Loan_Prediction.pkl`) for future predictions and integration.

## Results
The XGBoost model achieved the highest predictive accuracy, with notable precision and recall for predicting defaults. Feature importance analysis highlighted variables with the greatest impact on loan default likelihood.

## Conclusion
The project successfully developed a predictive model to assess loan default risk. Key insights from the EDA and feature importance analysis emphasize the importance of variables like income, credit history, and loan intent in assessing borrower risk.

## Files in the Repository
- **Loan_Prediction.ipynb**: Jupyter notebook with the complete code and analysis.
- **credit_risk_dataset.csv**: Dataset used for the analysis.
- **Loan_Prediction.pkl**: Serialized predictive model.

## Instructions for Running the Notebook
1. Clone this repository.
2. Ensure that the following libraries are installed:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn xgboost
