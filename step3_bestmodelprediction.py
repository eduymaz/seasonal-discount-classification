import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")


# Upload data / preprocess 
df = pd.read_csv("./data/processed_data.csv")
categorical_columns = ['product_name', 'category_name', 'yearquarter', 'city']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop(columns=['discount_effective', 'unit_price', 'order_date', 'customer_id', 'product_id', 'units_in_stock', 'category_id', 'year', 'month', 'day'])
y = df['discount_effective']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE (data standardization)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Min-Max Scaling (for KNN)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# --- MODELS TRAIN / TEST ---
results = []

# Decision Tree
dt_params = {"max_depth": [3, 5, 8, 10], "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 5, 10]}
dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=10, n_jobs=-1, verbose=0)
dt_model.fit(X_train_balanced, y_train_balanced)
best_dt = dt_model.best_estimator_
y_dt_prediction = best_dt.predict(X_test)
y_dt_probability = best_dt.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Decision Tree",
    "Accuracy": accuracy_score(y_test, y_dt_prediction),
    "Precision": precision_score(y_test, y_dt_prediction),
    "Recall": recall_score(y_test, y_dt_prediction),
    "F1-Score": f1_score(y_test, y_dt_prediction),
    "ROC-AUC": roc_auc_score(y_test, y_dt_probability)
    })

# KNN
knn_params = {"n_neighbors": np.arange(1, 50)}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=10, n_jobs=-1, verbose=0)
knn_model.fit(X_train_scaled, y_train_balanced)
best_knn = knn_model.best_estimator_
y_knn_prediction = best_knn.predict(X_test_scaled)
y_knn_probability = best_knn.predict_proba(X_test_scaled)[:, 1]

results.append({
    "Model": "KNN",
    "Accuracy": accuracy_score(y_test, y_knn_prediction),
    "Precision": precision_score(y_test, y_knn_prediction),
    "Recall": recall_score(y_test, y_knn_prediction),
    "F1-Score": f1_score(y_test, y_knn_prediction),
    "ROC-AUC": roc_auc_score(y_test, y_knn_probability)
})

# Logistic Regression
lr_params = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 500]}
lr_model = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
lr_model.fit(X_train_balanced, y_train_balanced)
best_lr = lr_model.best_estimator_
y_lr_prediction = best_lr.predict(X_test)
y_lr_probability = best_lr.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, y_lr_prediction),
    "Precision": precision_score(y_test, y_lr_prediction),
    "Recall": recall_score(y_test, y_lr_prediction),
    "F1-Score": f1_score(y_test, y_lr_prediction),
    "ROC-AUC": roc_auc_score(y_test, y_lr_probability)
})

# --- CONC TABLE ---
results_df = pd.DataFrame(results)
print(results_df)

# --- ROC  ---
plt.figure(figsize=(10, 6))
for model_name, y_probability in zip(
    ["Decision Tree", "KNN", "Logistic Regression"],
    [y_dt_probability, y_knn_probability, y_lr_probability]
):
    fpr, tpr, _ = roc_curve(y_test, y_probability)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_probability):.2f})")

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrileri")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./results/roc_curves.png")

# --- CONFUSION MATRIX GRAPHS ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, model_name, y_prediction in zip(
    axes,
    ["Decision Tree", "KNN", "Logistic Regression"],
    [y_dt_prediction, y_knn_prediction, y_lr_prediction]
):
    cm = confusion_matrix(y_test, y_prediction)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("./results/confusion_matrix.png")