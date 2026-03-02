# Diabetes Progression Prediction (Python)

## Project Overview
This project focuses on predicting patient glucose levels using clinical data (BMI, Age, Blood Pressure, etc.). It was developed as a final project for CSCI 1070 (Intro to Computer Science: Taming Big Data).

**Key Distinction:** To demonstrate a deep understanding of algorithmic logic, all models and data processing functions were built from scratch using core Python. This project avoids high-level libraries like Scikit-Learn or Pandas to focus on "under-the-hood" engineering.

## Technical Features
* **Manual Algorithm Implementation:** Built k-Nearest Neighbors (kNN) and Simple Linear Regression (SLR) models from the ground up.
* **Data Engineering Pipeline:** * Developed custom scripts to handle "zero-value" data gaps.
    * Implemented statistical mean imputation for missing feature values.
    * Engineered feature scaling and normalization for kNN accuracy.
* **Performance Optimization:** Applied the **Elbow Method** to programmatically determine the optimal $k$-value for regression accuracy.
* **Statistical Analysis:** Used Matplotlib to generate correlation matrices, scatter plots, and performance graphs (calculating RMSE and $R^2$ metrics).

## File Structure
* `diabetes_kNN_Regessions.py`: Main implementation of multi-feature kNN regression.
* `diabetes_SLR.py`: Simple Linear Regression focusing on BMI as a primary predictor.
* `evans_functions.py`: Custom utility module for data cleaning, scaling, and RMSE calculations.
* `diabetes.csv`: The clinical dataset used for model training and testing.

## Results & Conclusions
The analysis confirmed that while single features like BMI are correlated with glucose levels, a multi-feature kNN approach is significantly more effective.
* **kNN Regression:** $R^2 = 0.24$ | $RMSE = 25.11$
* **Linear Regression:** $R^2 = 0.06$ | $RMSE = 28.00$

The kNN model achieved a **300% improvement** in variance capture over the simple linear model, proving the necessity of multi-variable analysis in complex clinical datasets.

## How to Run
1. Ensure you have Python 3.x and Matplotlib installed.
2. Clone this repository.
3. Run `python diabetes_kNN_Regessions.py` to view model evaluation results and generated graphs.
