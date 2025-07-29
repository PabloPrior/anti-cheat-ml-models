import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import time
import itertools
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# ---------- CONFIG ----------
RESULTS_DIR = "SmoteResults"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- LOADING AND PREPROCESSING ----------
path_train = Path("../HAWK_AE/HAWK_AE/HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main/3. RevStats/dataset/final_train.csv")
df = pd.read_csv(path_train)

if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

X = df.drop(columns=["isCheater"])
y = df["isCheater"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)

X_train, X_valid, y_train, y_valid = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

selector = SelectKBest(score_func=f_classif, k=7)
X_train_best = selector.fit_transform(X_train, y_train)
X_valid_best = selector.transform(X_valid)


# ---------- AUX FUNCS ----------
def evaluate_model(model, params, name, results):
    try:
        model.set_params(**params)
        start = time.time()
        model.fit(X_train_best, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_valid_best)

        results.append({
            "Model": name,
            "Params": str(params),
            "Accuracy": round(accuracy_score(y_valid, y_pred), 4),
            "Precision": round(precision_score(y_valid, y_pred), 4),
            "Recall": round(recall_score(y_valid, y_pred), 4),
            "F1-Score": round(f1_score(y_valid, y_pred), 4),
            "Train time (s)": round(train_time, 2)
        })
    except Exception as e:
        print(f" Error with {name} y params={params}: {e}")


def manual_model(model_class, parameters, xTrain, xValid, yTrain, yValid, results):
    try:
        clf = model_class(**parameters)
        start = time.time()
        clf.fit(xTrain, yTrain)
        train_time = time.time() - start
        y_pred = clf.predict(xValid)

        results.append({
            "Model": model_class.__name__,
            "Parameters": str(parameters),
            "Accuracy": round(accuracy_score(yValid, y_pred), 4),
            "Precision": round(precision_score(yValid, y_pred), 4),
            "Recall": round(recall_score(yValid, y_pred), 4),
            "F1-Score": round(f1_score(yValid, y_pred), 4),
            "Train time (s)": round(train_time, 2)
        })
    except Exception as e:
        print(f"Error with {model_class.__name__} and params={parameters}: {e}")


#To generate all param combinations
def generate_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


# ---------- PARAMETERS ----------
param_grid_rf = {
    "n_estimators": [100, 300, 500, 1000],
    "max_depth": [5, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 10],
    "max_features": ['sqrt', 'log2', 0.2, 0.5, None],
    "bootstrap": [True],
    "criterion": ['gini', 'entropy']
}

param_grid_gb = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.5, 0.7, 1.0],
    "max_features": ['sqrt', 'log2', 0.2, 0.5, None],
    "loss": ['log_loss', 'exponential']
}

param_grid_xgb = {
    "n_estimators": [100, 300, 500, 1000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.5, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5, 1],
    "reg_lambda": [0, 1, 5, 10],
    "reg_alpha": [0, 1, 5, 10],
    "booster": ['gbtree'],
    "scale_pos_weight": [1]
}

# ---------- MANUAL GRIDS ----------
manual_grid_rf = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "class_weight": [None, "balanced"]
}

manual_grid_gb = {
    "n_estimators": [200, 500, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8]
}

manual_grid_xgb = {
    "n_estimators": [200, 500, 1000],
    "learning_rate": [0.005, 0.01, 0.05],
    "max_depth": [10, 15, 20],
    "scale_pos_weight": [1, 5, 10]
}


# ---------- RANDOM SEARCH ----------
print("RandomizedSearch with 60 iterations...")
results_random = []

search_rf = RandomizedSearchCV(RandomForestClassifier(), param_grid_rf, n_iter=60, scoring="f1", cv=3, random_state=42, n_jobs=-1)
search_rf.fit(X_train_best, y_train)
for params in search_rf.cv_results_["params"]:
    evaluate_model(RandomForestClassifier(), params, "RandomForest", results_random)
print("RF finished")
print()

search_gb = RandomizedSearchCV(GradientBoostingClassifier(), param_grid_gb, n_iter=60, scoring="f1", cv=3, random_state=42, n_jobs=-1)
search_gb.fit(X_train_best, y_train)
for params in search_gb.cv_results_["params"]:
    evaluate_model(GradientBoostingClassifier(), params, "GradientBoosting", results_random)
print("GB finished")
print()

search_xgb = RandomizedSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False), param_grid_xgb, n_iter=60, scoring="f1", cv=3, random_state=42, n_jobs=-1)
search_xgb.fit(X_train_best, y_train)
for params in search_xgb.cv_results_["params"]:
    evaluate_model(XGBClassifier(eval_metric='logloss', use_label_encoder=False), params, "XGBoost", results_random)
print("XGB finished")
print()

pd.DataFrame(results_random).to_excel(os.path.join(RESULTS_DIR, "RandomSearchSMOTE.xlsx"), index=False)
print("\n RandomSearch SMOTE finished. Results saved in Excel.")



# ---------- GRID SEARCH ----------
print("GridSearch...")
results_grid = []

#--- Random Forest ---
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring="f1", cv=3, n_jobs=-1, verbose=2)
grid_rf.fit(X_train_best, y_train)
for params in grid_rf.cv_results_["params"]:
    evaluate_model(RandomForestClassifier(), params, "RandomForest", results_grid)
    pd.DataFrame(results_grid).to_excel(os.path.join(RESULTS_DIR, "GridSearchRF.xlsx"), index=False)
print("RandomForest finished. Results saved.")

#--- Gradient Boosting ---
results_grid = []
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, scoring="f1", cv=3, n_jobs=-1, verbose=2)
grid_gb.fit(X_train_best, y_train)
for params in grid_gb.cv_results_["params"]:
    evaluate_model(GradientBoostingClassifier(), params, "GradientBoosting", results_grid)
    pd.DataFrame(results_grid).to_excel(os.path.join(RESULTS_DIR, "GridSearchGB.xlsx"), index=False)
print("GradientBoosting finished. Results saved.")
print()

# --- XGBoost ---
results_grid = []
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid_xgb,
                        scoring="f1", cv=3, n_jobs=-1, verbose=2)
grid_xgb.fit(X_train_best, y_train)
for params in grid_xgb.cv_results_["params"]:
    evaluate_model(XGBClassifier(eval_metric='logloss'), params, "XGBoost", results_grid)
    pd.DataFrame(results_grid).to_excel(os.path.join(RESULTS_DIR, "GridSearchXGB.xlsx"), index=False)
print("XGBoost finished. Results saved.")



# ---------- MANUAL EVALUATION ----------
results_manual_smote = []

print("Running RandomForest manual tuning with SMOTE...")
for params in generate_combinations(manual_grid_rf):
    manual_model(RandomForestClassifier, params, X_train_best, X_valid_best, y_train, y_valid, results_manual_smote)

print("Running GradientBoosting manual tuning with SMOTE...")
for params in generate_combinations(manual_grid_gb):
    manual_model(GradientBoostingClassifier, params, X_train_best, X_valid_best, y_train, y_valid, results_manual_smote)

print("Running XGBoost manual tuning with SMOTE...")
for params in generate_combinations(manual_grid_xgb):
    manual_model(XGBClassifier, params, X_train_best, X_valid_best, y_train, y_valid, results_manual_smote)

# ---------- SAVE ----------
pd.DataFrame(results_manual_smote).to_excel(os.path.join(RESULTS_DIR, "ManualSearchSMOTE.xlsx"), index=False)
print("\n Manual search with SMOTE finished. Results saved.")


