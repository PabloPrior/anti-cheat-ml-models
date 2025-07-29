from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

# ---------- CONFIG ----------
THRESHOLD = 0.3
PATH_TRAIN = Path("../HAWK_AE/HAWK_AE/HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main/3. RevStats/dataset/final_train.csv")
PATH_TEST = Path("../HAWK_AE/HAWK_AE/HAWK_Next-Gen_FPS_Anti-Cheat_Framework-main/3. RevStats/dataset/final_test.csv")

# ---------- LOAD DATA ----------
train_df = pd.read_csv(PATH_TRAIN)
test_df = pd.read_csv(PATH_TEST)

X_train = train_df.drop(columns=["ID", "isCheater"], errors='ignore')
y_train = train_df["isCheater"]

X_test = test_df.drop(columns=["ID", "isCheater"], errors='ignore')
y_test = test_df["isCheater"]

# ---------- SCALER AND SELECTOR ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=7)
X_train_best = selector.fit_transform(X_train_scaled, y_train)
X_test_best = selector.transform(X_test_scaled)

# ---------- BEST PARAMETERS MODEL ----------
best_params = {
    'subsample': 1.0,
    'n_estimators': 500,
    'min_samples_split': 2,
    'min_samples_leaf': 4,
    'max_features': None,
    'max_depth': 10,
    'loss': 'exponential',
    'learning_rate': 0.3
}

model = GradientBoostingClassifier(**best_params)
model.fit(X_train_best, y_train)

# ---------- PREDICT ----------
y_prob = model.predict_proba(X_test_best)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# ---------- EVALUATION ----------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n--- TEST EVALUATION (THRESHOLD = {THRESHOLD}) ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClasification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ---------- CURVA PRECISION-RECALL ----------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.axvline(THRESHOLD, color="red", linestyle="--", label=f"Threshold={THRESHOLD}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Curva Precision-Recall vs Threshold (GB)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
