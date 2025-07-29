# =============================================================================
# Project Title: Heart Disease Classification using SVM, Random Forest
# Dataset Source: Kaggle – Heart Disease Dataset by David Lapp
# =============================================================================
# Description:
# This project aims to classify the presence of heart disease using machine 
# learning models such as Support Vector Machine (SVM), Random Forest
# The dataset includes features like age, 
# cholesterol, blood pressure, and other medical indicators. Optuna is used 
# for hyperparameter optimization of each model individually.

# Author: Siddant.H
# =============================================================================

# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances,plot_parallel_coordinate

# Load dataset
df = pd.read_csv("c_heart1.csv")
print(df.info())
print(df.describe())
x = df.drop('target', axis=1)
y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training Models
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Define a function to print metrics
def print_metrics(y_true, y_pred, model_name):
    print(model_name)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print()
    
# Show metrics
print("SVM"," Metrics (Before Tuning)")
print_metrics(y_test, svm_pred, "SVM")
print("RF"," Metrics (Before Tuning)")
print_metrics(y_test, rf_pred, "Random Forest")

# Optuna for SVM
def objective_svm(trial):
    C = trial.suggest_float('C', 0.1, 100, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    return cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()

# Optuna for Random Forest
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

# Run study for SVM
study_svm = optuna.create_study(direction="maximize",study_name="For SVM")
study_svm.optimize(objective_svm, n_trials=30)
print("Best SVM params:", study_svm.best_params)

print()

# Run study for RF
study_rf = optuna.create_study(direction="maximize",study_name="For RF")
study_rf.optimize(objective_rf, n_trials=30)
print("Best RF params:", study_rf.best_params)


# SVM best model
best_C = study_svm.best_params['C']
best_kernel = study_svm.best_params['kernel']
best_gamma = study_svm.best_params['gamma']

best_svm = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma, random_state=42)
best_svm.fit(X_train_scaled, y_train)
svm_preds_tuned = best_svm.predict(X_test_scaled)

# Random Forest best model
best_n_estimators = study_rf.best_params['n_estimators']
best_max_depth = study_rf.best_params['max_depth']

best_rf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
best_rf.fit(X_train, y_train)
rf_preds_tuned = best_rf.predict(X_test)

# Print metrics
print_metrics(y_test, svm_preds_tuned, "SVM (After Tuning)")
print_metrics(y_test, rf_preds_tuned, "Random Forest (After Tuning)")


# -----------------------------
# ___OPTUNA VISUALIZATION___
# -----------------------------


# SVM Plots
plot_optimization_history(study_svm).show()
plot_parallel_coordinate(study_svm).show()
plot_param_importances(study_svm).show()

# Random Forest Plots
plot_optimization_history(study_rf).show()
plot_parallel_coordinate(study_rf).show()
plot_param_importances(study_rf).show()
