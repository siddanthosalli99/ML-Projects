# =============================================================================
# Project Title: Cervical Cancer ( Risk Factors )
# Dataset Source: Kaggle – Cervical Cancer Risk Classification by Gokagglers
# =============================================================================
# Description:
# This project aims to classify cervical cancer risk using machine 
# learning models such as Support Vector Machine (SVM), Random Forest, 
# Logistic Regression. The dataset includes medical risk factors 
# like age, sexual activity, smoking, and STDs. Data preprocessing includes 
# KNN imputation for missing values, outlier visualization, and MinMax scaling.
# Optuna is used for hyperparameter tuning of SVM and Random Forest.

# Author: Siddant.H
# =============================================================================

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

# Load dataset
df = pd.read_csv('c_cervical_cancer2.csv')
print(df.info())
print(df.describe())

# Replace ? with NaN and convert to numeric
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Checking for outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_imputed[['Number of sexual partners', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)']])
plt.title("Outlier Check")
plt.show()

# Separate features and target
x = df_imputed.drop(['Dx:Cancer'], axis=1)
y = df_imputed['Dx:Cancer']

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define metric function
def print_metrics(y_true, y_pred, model_name):
    print("\nResults for ", model_name)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))


# Optuna for Logistic Regression
def objective_logistic(trial):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

study_logistic = optuna.create_study(direction="maximize", study_name="Logistic Regression")
study_logistic.optimize(objective_logistic, n_trials=30)
best_params_log = study_logistic.best_params

model_log = LogisticRegression(
    C=best_params_log["C"],
    penalty=best_params_log["penalty"],
    solver=best_params_log["solver"],
    max_iter=1000
)
model_log.fit(X_train, y_train)
pred_log = model_log.predict(X_test)
print_metrics(y_test, pred_log, "Logistic Regression (Tuned)")


# Optuna for Random Forest
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

study_rf = optuna.create_study(direction="maximize", study_name="Random Forest")
study_rf.optimize(objective_rf, n_trials=30)
best_params_rf = study_rf.best_params

model_rf = RandomForestClassifier(
    n_estimators=best_params_rf["n_estimators"],
    max_depth=best_params_rf["max_depth"],
    random_state=42
)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
print_metrics(y_test, pred_rf, "Random Forest (Tuned)")


# Optuna for SVM
def objective_svm(trial):
    C = trial.suggest_float("C", 0.1, 100, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()

study_svm = optuna.create_study(direction="maximize", study_name="SVM")
study_svm.optimize(objective_svm, n_trials=30)
best_params_svm = study_svm.best_params

model_svm = SVC(
    C=best_params_svm["C"],
    kernel=best_params_svm["kernel"],
    gamma=best_params_svm["gamma"],
    random_state=42
)
model_svm.fit(X_train, y_train)
pred_svm = model_svm.predict(X_test)
print_metrics(y_test, pred_svm, "SVM (Tuned)")

# -----------------------------
# ___OPTUNA VISUALIZATION___
# -----------------------------


# Logistic Regression Plots
plot_optimization_history(study_logistic).show()
plot_parallel_coordinate(study_logistic).show()
plot_param_importances(study_logistic).show()

# Random Forest Plots
plot_optimization_history(study_rf).show()
plot_parallel_coordinate(study_rf).show()
plot_param_importances(study_rf).show()

# SVM Plots
plot_optimization_history(study_svm).show()
plot_parallel_coordinate(study_svm).show()
plot_param_importances(study_svm).show()
