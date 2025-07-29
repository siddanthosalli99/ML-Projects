# =============================================================================
# Project Title: Insurance Cost Prediction using Random Forest & XGBoost
# Source of the CSV File: Kaggle - Medical Cost Personal Datasets by Miri Choi
# =============================================================================
# Description:
# This project predicts medical insurance charges using features like age, BMI,
# smoking habits, and more, using regression tree models (Random Forest & XGBoost).
# This dataset contains both numerical and categorial features
# -----------------------------------------------------------------------------

# Author: Siddant.H
# =============================================================================

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_param_importances,plot_parallel_coordinate,plot_contour

# Load data
df = pd.read_csv('r_insurance3.csv')
print(df.head())

# Basic EDA
print(df.info())
print(df.describe())

# Encoding & Scaling
categorical = ['sex', 'smoker', 'region']
numerical = ['age', 'bmi', 'children']
target = 'charges'

# Preprocessing using column transformer
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first'), categorical),
    ('scale', MinMaxScaler(), numerical)
])

x = df.drop(columns=target)
y = df[target]

X_new = preprocessor.fit_transform(x)
feature_names = preprocessor.get_feature_names_out()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=172, max_depth=4, min_samples_split=3, min_samples_leaf=3, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=167, learning_rate= 0.01, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, name):
    print(name)
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R² Score:", r2_score(y_true, y_pred))

print()
evaluate(y_test, rf_pred, "Random Forest")
print()
evaluate(y_test, xgb_pred, "XGBoost")

# ----------------------
# ____CROSS VALIDATION____
# ----------------------

rf_cv = cross_val_score(rf, X_new, y, cv=5, scoring='r2')
xgb_cv = cross_val_score(xgb, X_new, y, cv=5, scoring='r2')
print("\nRandom Forest CV R² Mean:", rf_cv.mean())
print("XGBoost CV R² Mean:", xgb_cv.mean())

# ----------------------
# ____OPTUNA TUNING____
# ----------------------

# Random Forest Optuna
def rf_objective(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 4, 15),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        random_state=42
    )
    return cross_val_score(model, X_new, y, cv=5, scoring='r2').mean()

rf_study = optuna.create_study(direction='maximize', study_name="RandomForest Optimization")
rf_study.optimize(rf_objective, n_trials=30)
print("Best RF Params:", rf_study.best_params)

print()
print()

# XGBoost Optuna
def xgb_objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=42
    )
    return cross_val_score(model, X_new, y, cv=5, scoring='r2').mean()

xgb_study = optuna.create_study(direction='maximize', study_name="XGBoost Optimization")
xgb_study.optimize(xgb_objective, n_trials=30)
print("Best XGBoost Params:", xgb_study.best_params)

# -----------------------------
# ___OPTUNA VISUALIZATION___
# -----------------------------


# Random Forest Plots
plot_optimization_history(rf_study).show()
plot_slice(rf_study).show()
plot_parallel_coordinate(rf_study).show()
plot_param_importances(rf_study).show()
plot_contour(rf_study).show()

# XGBoost Plots
plot_optimization_history(xgb_study).show()
plot_slice(xgb_study).show()
plot_parallel_coordinate(xgb_study).show()
plot_param_importances(xgb_study).show()
plot_contour(xgb_study).show()
