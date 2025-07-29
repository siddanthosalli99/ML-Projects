# =============================================================================
# Project Title: Air Quality Index Prediction using Random Forest & XGBoost
# Source of the CSV File: Kaggle - Air Quality Data in India (2015 - 2020) by Vopani
# =============================================================================
# Description:
# This project aims to predict the Air Quality Index (AQI) using machine learning
# regression models on a real-world dataset containing atmospheric pollutant levels.

# Author: Siddant.H
# =============================================================================


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances

# Load data
df = pd.read_csv('r_city_day1.csv')
df = df.drop(df.columns[[0, 1, 15]], axis=1)

# Basic EDA
print(df.info())
print(df.describe())

# Handle missing values
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Scaling
scaler = MinMaxScaler()
x = df_imputed.drop('AQI', axis=1)
y = df_imputed['AQI']
X_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=291,max_depth=13,min_samples_split=8,min_samples_leaf=3, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=289, learning_rate=0.04017549818163242, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, name):
    print(name)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R² Score:", r2_score(y_true, y_pred))
    print()

evaluate(y_test, rf_pred, "Random Forest")
evaluate(y_test, xgb_pred, "XGBoost")


# ----------------------
# ____CROSS VALIDATION____
# ----------------------

rf_cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='r2')
xgb_cv_scores = cross_val_score(xgb, X_scaled, y, cv=5, scoring='r2')

print("Random Forest CV R² Mean:", rf_cv_scores.mean())
print("XGBoost CV R² Mean:", xgb_cv_scores.mean())
print()

# ----------------------
# ____OPTUNA TUNING____
# ----------------------

# Random Forest Optuna
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    score = cross_val_score(model, X_scaled, y, cv=3, scoring='r2').mean()
    return score

rf_study = optuna.create_study(direction='maximize',study_name="Random Forest")
rf_study.optimize(rf_objective, n_trials=30)
print("Best RF Params:", rf_study.best_params)

# XGBoost Optuna
def xgb_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42
    )

    score = cross_val_score(model, X_scaled, y, cv=3, scoring='r2').mean()
    return score

xgb_study = optuna.create_study(direction='maximize')
xgb_study.optimize(xgb_objective, n_trials=30)
print("Best XGBoost Params:", xgb_study.best_params)

#-----------------------------
#___ OPTUNA VISUALIAZATION ___
#-----------------------------

# Random Forest Optimization History
plot_optimization_history(rf_study).show()
# Random Forest Slice Plot
plot_slice(rf_study).show()
# Random Forest Hyperparameter Importance
plot_param_importances(rf_study).show()


# XGBoost Optimization History
plot_optimization_history(xgb_study).show()
# XGBoost Slice Plot
plot_slice(xgb_study).show()
# XGBoost Hyperparameter Importance
plot_param_importances(xgb_study).show()
