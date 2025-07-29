import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# -------- FILES --------
train_path = Path("../HAWK_AE/HAWK_AE/HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main/3. RevStats/dataset/final_train.csv")
test_path = Path("../HAWK_AE/HAWK_AE/HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main/3. RevStats/dataset/final_test.csv")

# -------- LOAD DATA --------
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Drop ID column if exists
for df in [df_train, df_test]:
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

# Split into X and y
X_train_raw = df_train.drop(columns=["isCheater"])
y_train = df_train["isCheater"]
X_test_raw = df_test.drop(columns=["isCheater"])
y_test = df_test["isCheater"]

# -------- PREPROCESSING --------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train_scaled, y_train)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=7)
X_train_best = selector.fit_transform(X_bal, y_bal)
X_test_best = selector.transform(X_test_scaled)

# -------- MODEL TRAINING --------
params = {
    'subsample': 1.0,
    'n_estimators': 500,
    'min_samples_split': 2,
    'min_samples_leaf': 4,
    'max_features': None,
    'max_depth': 10,
    'loss': 'exponential',
    'learning_rate': 0.3
}

model = GradientBoostingClassifier(**params)
model.fit(X_train_best, y_bal)

# -------- EVALUATION --------
y_pred = model.predict(X_test_best)

print("\n--- Evaluation on TEST SET ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

print()
print(confusion_matrix(y_test, y_pred))
