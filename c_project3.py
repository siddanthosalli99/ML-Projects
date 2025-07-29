# =============================================================================
# Project Title: Pulsar Star Classification using Logistic Regression, SVM, Random Forest
# Source of the CSV File: Kaggle - Pulsar Classification For Class Prediction by Baris Dincer
# =============================================================================
# Description:
# This project aims to classify whether a star is a pulsar using supervised machine learning
# on a real-world dataset with signal measurements from the HTRU2 dataset.

# Author: Siddant.H
# =============================================================================


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate


# LOAD DATA
df = pd.read_csv('c_pulsar3.csv')
df.rename(columns={'target_class': 'Class'}, inplace=True)

# CLEANING & EDA
print(df.info())
print(df.describe())

# Remove outliers using IQR
def remove_outliers_iqr(data, features):
    for col in features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df = remove_outliers_iqr(df, df.columns[:-1])

# FEATURE SCALING
x = df.drop('Class', axis=1)
y = df['Class']
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# OPTUNA TUNING

# Logistic Regression
def logreg_objective(trial):
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    model = LogisticRegression(C=C, max_iter=1000)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

logreg_study = optuna.create_study(direction="maximize",study_name="For Logistic Regression")
logreg_study.optimize(logreg_objective, n_trials=30)
print("Best Logistic Regression Params:", logreg_study.best_params)
print()
print()

# SVM
def svm_objective(trial):
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

svm_study = optuna.create_study(direction="maximize",study_name="For SVM")
svm_study.optimize(svm_objective, n_trials=30)
print("Best SVM Params:", svm_study.best_params)
print()
print()

# Random Forest
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion=criterion,
        random_state=42
    )

    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

rf_study = optuna.create_study(direction="maximize",study_name="For Random Forest")
rf_study.optimize(rf_objective, n_trials=30)
print("Best Random Forest Params:", rf_study.best_params)
print()
print()

# FINAL MODELS

# Logistic Regression
logreg = LogisticRegression(C=logreg_study.best_params['C'], max_iter=1000)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

# SVM
svm = SVC(
    C=svm_study.best_params['C'],
    kernel=svm_study.best_params['kernel'],
    gamma=svm_study.best_params['gamma']
)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=rf_study.best_params['n_estimators'],
    max_depth=rf_study.best_params['max_depth'],
    criterion=rf_study.best_params['criterion'],
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# EVALUATION
import warnings
warnings.filterwarnings("ignore")
def evaluate(y_true, y_pred, name):
    print(name)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print()
    print()
evaluate(y_test, logreg_pred, "Logistic Regression")
evaluate(y_test, svm_pred, "SVM")
evaluate(y_test, rf_pred, "Random Forest")

# CROSS VALIDATION

logreg_cv = cross_val_score(logreg, X_scaled, y, cv=5, scoring='accuracy')
svm_cv = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
rf_cv = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')

print("Logistic Regression CV Accuracy Mean:", logreg_cv.mean())
print("SVM CV Accuracy Mean:", svm_cv.mean())
print("Random Forest CV Accuracy Mean:", rf_cv.mean())


# Logistic Regression Plots
plot_optimization_history(logreg_study).show()
plot_parallel_coordinate(logreg_study).show()
plot_param_importances(logreg_study).show()

# SVM Plots
plot_optimization_history(svm_study).show()
plot_parallel_coordinate(svm_study).show()
plot_param_importances(svm_study).show()

# Random Forest Plots
plot_optimization_history(rf_study).show()
plot_parallel_coordinate(rf_study).show()
plot_param_importances(rf_study).show()