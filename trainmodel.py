import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import matplotlib.pyplot as plt
from collections import Counter

# Create directories for output
os.makedirs("evaluation", exist_ok=True)
os.makedirs("evaluation/figures", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load your dataset - FIXED: Changed to dataset.csv
df = pd.read_csv("final_dataset.csv")
df = df.dropna(axis=1, how='all')
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

print("Original DataFrame columns:", df.columns.tolist())

# Convert month to numeric and fill missing values with the median
df["Month"] = pd.to_numeric(df["Month"], errors='coerce')
month_median = df["Month"].median()
df["Month"] = df["Month"].fillna(month_median)
print(f"Converted Month to numeric, filled missing values with median: {month_median}")

# Create encoding for categorical columns
encoder_domain = LabelEncoder()
encoder_level = LabelEncoder()
encoder_skill = LabelEncoder()
encoder_resource1 = LabelEncoder()
encoder_resource2 = LabelEncoder()
encoder_project = LabelEncoder()

# Perform encoding and CREATE the new columns
df["domain_encoded"] = encoder_domain.fit_transform(df["Domain"].fillna("Unknown"))
df["level_encoded"] = encoder_level.fit_transform(df["Level"].fillna("Unknown"))
df["skill_encoded"] = encoder_skill.fit_transform(df["Skill"].fillna("Unknown"))
df["resource1_encoded"] = encoder_resource1.fit_transform(df["Recommended Resource 1"].fillna("Unknown"))
df["resource2_encoded"] = encoder_resource2.fit_transform(df["Recommended Resource 2"].fillna("Unknown"))
df["project_encoded"] = encoder_project.fit_transform(df["Project"].fillna("Unknown"))

print("DataFrame columns after encoding:", df.columns.tolist())

# Check class distribution
skill_counts = Counter(df["skill_encoded"])
print("\n===== CLASS DISTRIBUTION =====")
print(f"Skill classes count: {skill_counts}")
print(f"Minimum samples per class: {min(skill_counts.values())}")
print(f"Number of classes with only 1 sample: {sum(1 for count in skill_counts.values() if count == 1)}")

# Define features & targets for skill prediction
X_skill = df[["domain_encoded", "Month", "level_encoded"]]
y_skill = df["skill_encoded"]

# Define features & targets for resource and project prediction
X_resource_project = df[["domain_encoded", "Month", "level_encoded"]]
y_resource1 = df["resource1_encoded"]
y_resource2 = df["resource2_encoded"]
y_project = df["project_encoded"]

# Remove stratify parameter since many classes have only 1 sample
X_train, X_test, y_train, y_test = train_test_split(X_skill, y_skill, test_size=0.2, random_state=42)
print("\nSuccessfully split the data without stratification")

df.to_csv("preprocessed_dataset.csv", index=False)
print("Preprocessed dataset saved as preprocessed_dataset.csv")

# Train multiple models - ADDED: Decision Tree and KNN models
print("\n===== TRAINING MODELS =====")

# 1. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully")

# 2. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree model trained successfully")

# 3. KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("KNN model trained successfully")

# Train resource and project recommendation models
resource1_model = RandomForestClassifier(n_estimators=100, random_state=42)
resource1_model.fit(X_resource_project, y_resource1)
print("Resource 1 recommendation model trained successfully")

resource2_model = RandomForestClassifier(n_estimators=100, random_state=42)
resource2_model.fit(X_resource_project, y_resource2)
print("Resource 2 recommendation model trained successfully")

project_model = RandomForestClassifier(n_estimators=100, random_state=42)
project_model.fit(X_resource_project, y_project)
print("Project recommendation model trained successfully")

# Save models
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(dt_model, "models/decision_tree_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")
joblib.dump(resource1_model, "models/resource1_recommendation_model.pkl")
joblib.dump(resource2_model, "models/resource2_recommendation_model.pkl")
joblib.dump(project_model, "models/project_recommendation_model.pkl")
print("All models saved to models/ directory")

# Save encoders for future use
joblib.dump(encoder_domain, "encoders/encoder_domain.pkl")
joblib.dump(encoder_level, "encoders/encoder_level.pkl")
joblib.dump(encoder_skill, "encoders/encoder_skill.pkl")
joblib.dump(encoder_resource1, "encoders/encoder_resource1.pkl")
joblib.dump(encoder_resource2, "encoders/encoder_resource2.pkl")
joblib.dump(encoder_project, "encoders/encoder_project.pkl")
print("Encoders saved to encoders/ directory")

# EVALUATION: Test accuracy for all models
print("\n===== TEST SET ACCURACY =====")
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
print(f"Decision Tree Test Accuracy: {dt_accuracy:.4f}")
print(f"KNN Test Accuracy: {knn_accuracy:.4f}")

# Cross-validation - Use KFold instead of StratifiedKFold due to class imbalance
print("\n===== CROSS-VALIDATION RESULTS =====")
# Use regular KFold instead of StratifiedKFold due to class imbalance
n_folds = min(5, len(df))
print(f"Using {n_folds} cross-validation folds")

try:
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf_model, X_skill, y_skill, cv=cv)
    dt_cv_scores = cross_val_score(dt_model, X_skill, y_skill, cv=cv)
    knn_cv_scores = cross_val_score(knn_model, X_skill, y_skill, cv=cv)
    
    print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    print(f"Decision Tree CV Accuracy: {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}")
    print(f"KNN CV Accuracy: {knn_cv_scores.mean():.4f} ± {knn_cv_scores.std():.4f}")
except Exception as e:
    print(f"Cross-validation error: {str(e)}")
    print("Skipping cross-validation due to class imbalance issues")

# Feature Importance for Random Forest
print("\n===== FEATURE IMPORTANCE =====")
feature_names = X_skill.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Create feature importance plot
try:
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for Skill Recommendation")
    plt.bar(range(X_skill.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_skill.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('evaluation/figures/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to evaluation/figures/feature_importance.png")
except Exception as e:
    print(f"Error generating feature importance plot: {str(e)}")

print("\nEvaluation complete! Models and encoders are ready for the Streamlit app.")