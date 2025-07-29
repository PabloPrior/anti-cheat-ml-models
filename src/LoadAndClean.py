import itertools

import pandas as pd
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from sklearn.feature_selection import SelectKBest, chi2, f_classif

from time import time

from openpyxl.workbook import Workbook

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.preprocessing import MinMaxScaler

basePath = (Path("..") / r"HAWK_AE\HAWK_AE\HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main".
            replace("\\", "/"))

datasets = {
    "2.RevPOV": [],
    "3.RevStats": [],
    "5.MVIN": []
}

### LOAD 2.RevPOV###
datasets["2.RevPOV"].append(os.path.join(basePath, r"2. RevPOV\Dataset\merged_output.csv"))
datasets["2.RevPOV"].append(os.path.join(basePath, r"2. RevPOV\Dataset\test.csv"))
datasets["2.RevPOV"].append(os.path.join(basePath, r"2. RevPOV\Dataset\val.csv"))

### LOAD 3.RevStats###
datasets["3.RevStats"].append(os.path.join(basePath, r"3. RevStats\dataset\final_train.csv"))
datasets["3.RevStats"].append(os.path.join(basePath, r"3. RevStats\dataset\final_val.csv"))
datasets["3.RevStats"].append(os.path.join(basePath, r"3. RevStats\dataset\final_test.csv"))

### LOAD 3.RevStats###
datasets["5.MVIN"].append(os.path.join(basePath, r"5. MVIN\dataset\ExSPC_predictions.csv"))
datasets["5.MVIN"].append(os.path.join(basePath, r"5. MVIN\dataset\RevPOV_predictions.csv"))
datasets["5.MVIN"].append(os.path.join(basePath, r"5. MVIN\dataset\RevStats_predictions.csv"))


def load_and_merge(files):
    dataframes = [pd.read_csv(file) for file in files]
    return pd.concat(dataframes, ignore_index=True)

def checkInfo(index):
    df = pd.read_csv(datasets[index][1])
    print(df)


def checkClean(r):
    ruta = (Path("..") / r.replace("\\", "/"))

    df = pd.read_csv(ruta)
    # print(df)
    # print(df.nunique())
    print(df.isnull().sum())
    print()
    print(df.duplicated().sum())
    print()
    print(df.dtypes)
    # print()
    # print(df['ID'] == len(df))
    # print()
    # print(df.describe())


def loadDatasets(dictionary):
    dataframes = {}

    for category, files in dictionary.items():
        dataframes[category] = {}  #sub dictionaries per category
        for file in files:
            fileName = file.split("\\")[-1]  #name file
            df = pd.read_csv(file)  #load with pd

            # Delete ID column
            if "ID" in df.columns:
                df.drop(columns=["ID"], inplace=True)

            dataframes[category][fileName] = df  #save dictionary
            #print(f"Loaded: {fileName} ({df.shape[0]} files, {df.shape[1]} columns)")
        print()

    return dataframes


def prepareData(df):
    target = "isCheater"
    #testSize = 0.2

    #Separate predictor variables "x" and target variable "y"
    x = df.drop(columns=[target]) #delete target variable
    y = df[target] #target variable (0 = clean, 1 = cheater)

    #Divide into train and test
    #xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=testSize, random_state=42, stratify=y)
    xTrain, xTemp, yTrain, yTemp = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
    xValid, xTest, yValid, yTest = train_test_split(xTemp, yTemp, test_size=0.5, random_state=42, stratify=yTemp)

    #Scale data
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xValid = scaler.transform(xValid)
    xTest = scaler.transform(xTest)

    print(f"Prepared data: Train={len(xTrain)}, Valid={len(xValid)}, Test={len(xTest)}")

    return xTrain, xValid, xTest, yTrain, yValid, yTest


def trainAndTest(model, parameters, xTrain, xValid, yTrain, yValid, results):

    #Initialize model with parameters
    clf = model(**parameters)

    #Prediction time
    start = time()
    clf.fit(xTrain, yTrain)
    trainTime = time() - start

    #Make predictions
    yPred = clf.predict(xValid)

    #Evaluate results
    accuracy = accuracy_score(yValid, yPred)
    precision = precision_score(yValid, yPred)
    recall = recall_score(yValid, yPred)
    f1 = f1_score(yValid, yPred)

    #Save results
    results.append({
        "Model": model.__name__,
        "Parameters": str(parameters),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Train time (s)": round(trainTime, 2)
    })


#Load datasets
dataframes = loadDatasets(datasets)
df = dataframes["3.RevStats"]["final_train.csv"]

#Preparare data
xTrain, xValid, xTest, yTrain, yValid, yTest = prepareData(df)

#7 best carac
kBest = SelectKBest(score_func=f_classif, k=7)
xTrainBest = kBest.fit_transform(xTrain, yTrain)
xValidBest = kBest.transform(xValid)

#Configs
param_grid_rf = {
    "n_estimators": [100, 300, 500, 1000, 1500, 2000],
    "max_depth": [5, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced", None]
}

param_grid_gb = {
    "n_estimators": [100, 300, 500, 700, 1000],
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
    "max_depth": [4, 6, 8, 12, 16]
}

param_grid_xgb = {
    "n_estimators": [300, 500, 700, 1000, 1500],
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
    "max_depth": [10, 15, 20, 25, 30],
    "scale_pos_weight": [1, 3, 5, 10, 20, 50]
}

#MANUAL
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


param_CV_rf = {
    "n_estimators": [100, 300, 500],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced", None]
}

param_CV_gb = {
    "n_estimators": [100, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 8, 12]
}

param_CV_xgb = {
    "n_estimators": [300, 500, 700],
    "learning_rate": [0.005, 0.01, 0.05],
    "max_depth": [10, 20, 30],
    "scale_pos_weight": [1, 5, 10]
}


def generate_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


#To find best hiperparameters
def tune_and_train(model, param_grid, x_train, y_train):
    search = RandomizedSearchCV(
        model(),
        param_distributions=param_grid,
        n_iter=60,  #Combinations
        cv=5,  #Cross validation
        scoring="accuracy",
        n_jobs=-1,  #Parallelization
        random_state=42,
        verbose=1
    )

    search.fit(x_train, y_train)
    return search, search.best_estimator_, search.best_params_


#Train and optimize models
search_rf, best_rf, best_params_rf = tune_and_train(RandomForestClassifier, param_grid_rf, xTrainBest, yTrain)
search_gb, best_gb, best_params_gb = tune_and_train(GradientBoostingClassifier, param_grid_gb, xTrainBest, yTrain)
search_xgb, best_xgb, best_params_xgb = tune_and_train(XGBClassifier, param_grid_xgb, xTrainBest, yTrain)

print("\nBest search params:")
print("Random Forest:", best_params_rf)
print("Gradient Boosting:", best_params_gb)
print("XGBoost:", best_params_xgb)


# ---------- RANDOM SEARCH ----------
#Evaluate models with all combinations of RandomizedSearchCV
results = []

for search, model, name in [(search_rf, best_rf, "RandomForest"),
                            (search_gb, best_gb, "GradientBoosting"),
                            (search_xgb, best_xgb, "XGBoost")]:

    print(f"\n Evaluating all combinations of {name}...")

    cv_results = search.cv_results_

    for i in range(len(cv_results["params"])):
        params = cv_results["params"][i]  #Hyperparameters used in this iteration

        #Assign hyperparameters to the model and train it
        model.set_params(**params)

        start = time()
        model.fit(xTrainBest, yTrain)
        train_time = time() - start

        #Obtain probabilities to apply the threshold
        y_probs = model.predict_proba(xValidBest)[:, 1]

        #Apply the decision threshold
        umbral = 0.4
        y_pred = (y_probs >= umbral).astype(int)

        #Calculate metrics
        accuracy = accuracy_score(yValid, y_pred)
        precision = precision_score(yValid, y_pred)
        recall = recall_score(yValid, y_pred)
        f1 = f1_score(yValid, y_pred)

        #Save results with the same structure
        results.append({
            "Model": name,
            "Params": str(params),
            "Mean Accuracy (CV)": cv_results["mean_test_score"][i],
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Train time (s)": round(train_time, 2)
        })

    print(f" {name} finished. Total combinations evaluated: {len(cv_results['params'])}")

#Save in Excel
os.makedirs("Results", exist_ok=True)
dfResults = pd.DataFrame(results)
dfResults.to_excel("Results/RandomizedSearchAllResults1(Umbral 0.4).xlsx", index=False)

print("\n Saved results in 'Results/RandomizedSearchAllResults1(Umbral 0.4).xlsx'")


# ---------- MANUAL SEARCH ----------
rf_combinations = generate_combinations(param_grid_rf)
gb_combinations = generate_combinations(param_grid_gb)
xgb_combinations = generate_combinations(param_grid_xgb)

results_manual = []

for params in rf_combinations:
    trainAndTest(RandomForestClassifier, params, xTrainBest, xValidBest, yTrain, yValid, results_manual)
print("RF finished")

for params in gb_combinations:
    trainAndTest(GradientBoostingClassifier, params, xTrainBest, xValidBest, yTrain, yValid, results_manual)
print("GB finished")

for params in xgb_combinations:
    trainAndTest(XGBClassifier, params, xTrainBest, xValidBest, yTrain, yValid, results_manual)
print("XGB finished")

#Save in Excel
dfResultsManual = pd.DataFrame(results_manual)
dfResultsManual.to_excel("Results/ManualResultsCombinationsExtended.xlsx", index=False)

print("\n Saved results in 'Results/ManualResultsCombinationsExtended.xlsx'")


# ---------- GRID SEARCH ----------
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
xgb = XGBClassifier()

models_param_grids = {
    "RandomForest": (RandomForestClassifier(), param_grid_rf),
    "GradientBoosting": (GradientBoostingClassifier(), param_grid_gb),
    "XGBoost": (XGBClassifier(), param_grid_xgb)
}

results = []

#Evaluate with multiple metrics
scoring_metrics = ["accuracy", "precision", "recall", "f1"]

for model_name, (model, param_grid) in models_param_grids.items():
    print(f"\n Running GridSearch for {model_name}...")

    grid_search = GridSearchCV(
        model, param_grid, scoring=scoring_metrics, refit="accuracy",
        cv=5, n_jobs=-1, verbose=2
    )

    grid_search.fit(xTrainBest, yTrain)

    #Get all tested combinations
    cv_results = grid_search.cv_results_

    for i in range(len(cv_results["params"])):
        params = cv_results["params"][i]  #Hyperparameters used in this iteration

        #Train the model with these parameters
        model.set_params(**params)
        model.fit(xTrainBest, yTrain)

        #Prediction on validation set
        y_pred = model.predict(xValidBest)

        #Calculate metrics in validation
        accuracy = accuracy_score(yValid, y_pred)
        precision = precision_score(yValid, y_pred)
        recall = recall_score(yValid, y_pred)
        f1 = f1_score(yValid, y_pred)

        #Save results maintaining the structure
        results.append({
            "Model": model_name,
            "Params": str(params),
            "Mean Accuracy (CV)": cv_results["mean_test_accuracy"][i],
            "Accuracy (Validation)": accuracy,
            "Precision (Validation)": precision,
            "Recall (Validation)": recall,
            "F1-Score (Validation)": f1
        })

    print(f" {model_name} finished. Total combinations evaluated: {len(cv_results['params'])}")

#Saved in Excel
df_results = pd.DataFrame(results)
df_results.to_excel("Results/GridSearchAllResults.xlsx", index=False)

print("\n Saved results in 'Results/GridSearchAllResults.xlsx'")

print()